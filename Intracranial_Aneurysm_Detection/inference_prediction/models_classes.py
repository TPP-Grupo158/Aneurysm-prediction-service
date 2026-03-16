import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage.transform import resize
from typing import Tuple, Union, List, Tuple, Dict, Any
import numpy as np
import os
import nibabel as nib

# ==================================================
# Plane Classification Model Definition
# ==================================================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64
        self.block = block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.embed_dim = 512 * block.expansion
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(query_num * embed_dim, num_classes) 
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        for name, param in self.cross_attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)
        
        attended, _ = self.cross_attention(query=query, key=x, value=x)
        
        attended = self.norm(attended)
        attended = self.dropout(attended)
        attended_permuted = attended.permute(1, 0, 2)
        attended_flatten = attended_permuted.flatten(1)
        logits = self.classifier(attended_flatten) 
        return logits


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, dropout=0.0, use_cross_attention=True, num_heads=4):
        super(ClassificationHead, self).__init__()
        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim=embed_dim, 
                query_num=query_num, 
                num_classes=num_classes, 
                num_heads=num_heads, 
                dropout=dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(1), 
                nn.Dropout(dropout), 
                nn.Linear(embed_dim, num_classes)
            )
    
    def forward(self, x):
        return self.pooling(x)


class PlaneResNet34(nn.Module):
    """Single-task model: Plane classification only (3 classes: AX/SAG/COR)"""
    
    def __init__(self, dropout: float = 0.1):
        super(PlaneResNet34, self).__init__()
        
        self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3], in_channels=1)
        self.embed_dim = self.encoder.embed_dim
        
        self.head_plane = ClassificationHead(
            embed_dim=self.embed_dim, 
            query_num=3, 
            num_classes=3, 
            dropout=dropout, 
            use_cross_attention=True
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head_plane(features)
        return logits

# ==================================================
# Plane Classification Predictor
# ==================================================



class PlaneClassifier:
    """Axial Slice Plane Prediction"""
    
    # Plane Category Mapping
    PLANE_MAP = {0: 'AX', 1: 'SAG', 2: 'COR'}
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', target_size=(256, 256)):
        """
        Initialize the inferencer.
        
        Args:
            checkpoint_path: Path to the model weights.
            device: Inference device (e.g., 'cuda', 'cpu').
            target_size: Target image dimensions.
        """
        self.device = device
        self.target_size = target_size
        self.model = self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str) -> PlaneResNet34:
        """Load model - Modified: Using PlaneResNet34"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file does not exist: {checkpoint_path}")
        
        model = PlaneResNet34(dropout=0.0)
        
        # Load model weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """Preprocesses a 2D slice."""
        # 1. Type conversion and clipping
        slice_data = slice_2d
        
        # 2. Z-score normalization
        mean = slice_data.mean()
        std = slice_data.std()
        std = np.clip(std, 1e-8, None)
        slice_data = (slice_data - mean) / std
        
        # 3. Resize
        resized_slice = resize(slice_data, self.target_size, anti_aliasing=True).astype(np.float32)
        
        # 4. Convert to Tensor
        tensor = torch.from_numpy(resized_slice).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def predict(self, tensor: torch.Tensor) -> Tuple[str, int, float, torch.Tensor]:
        """
        Performs Plane prediction - Updated: adapted for single-task model output format
        
        Args:
            tensor: Input tensor of shape [1, 1, H, W]
            
        Returns:
            plane_pred_label: Predicted class name ('AX', 'SAG', 'COR')
            plane_pred: Predicted class index (0, 1, 2)
            plane_prob: Maximum probability value
            plane_prob_list: Probability distribution over all classes [1, 3]
        """
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
        
        # Plane prediction
        plane_pred = logits.argmax(dim=1).item()
        plane_prob_list = F.softmax(logits, dim=1)
        plane_prob = plane_prob_list.max().item()
        plane_pred_label = self.PLANE_MAP[plane_pred]
        
        return plane_pred_label, plane_pred, plane_prob, plane_prob_list
    
    def inference_from_slice(
        self, 
        slice_2d: np.ndarray,
    ) -> Tuple[str, int, float, torch.Tensor]:
        """
        Performs inference from a 2D slice.
        
        Returns:
            plane_pred_label: Predicted class name
            plane_pred: Predicted class index
            plane_prob: Maximum probability
            plane_prob_list: Probability distribution
        """
        # 1. Preprocessing
        tensor = self.preprocess_slice(slice_2d)
        
        # 2. Inference
        plane_pred_label, plane_pred, plane_prob, plane_prob_list = self.predict(tensor)
        
        # 3. Print results
        self._print_result(plane_pred_label, plane_pred, plane_prob, plane_prob_list)
        
        return plane_pred_label, plane_pred, plane_prob, plane_prob_list
    
    def _print_result(self, plane_pred_label: str, plane_pred: int, plane_prob: float, plane_prob_list: torch.Tensor):
        """Prints the prediction results."""
        print("=" * 60)
        print("Plane Prediction Results:")
        print(f"  Predicted Class: {plane_pred_label}")
        print(f"  Class Index: {plane_pred}")
        print(f"  Confidence: {plane_prob:.4f} ({plane_prob*100:.2f}%)")
        print(f"  Probability Distribution: {plane_prob_list}")
        print("=" * 60)

def correct_orientation(pixel_array, spacing, plane_id):
    """
    Corrects image orientation to standard axial view.
    """
    if plane_id == 1:
        # Sagittal → Axial
        fixed_array = np.transpose(pixel_array, (1, 2, 0))
        fixed_array = fixed_array[::-1, :, :]
        fixed_spacing = [spacing[1], spacing[2], spacing[0]]
        print(f"  Corrected: {pixel_array.shape} → {fixed_array.shape}")
        print(f"  Corrected: {spacing} → {fixed_spacing}")
        return fixed_array, fixed_spacing

    elif plane_id == 2:
        # Coronal → Axial
        fixed_array = np.transpose(pixel_array, (1, 0, 2))
        fixed_array = fixed_array[::-1, :, :]
        fixed_spacing = [spacing[1], spacing[0], spacing[2]]
        print(f"  Corrected: {pixel_array.shape} → {fixed_array.shape}")
        print(f"  Corrected: {spacing} → {fixed_spacing}")
        return fixed_array, fixed_spacing
    else:
        # Already axial or no correction needed
        return pixel_array, spacing


def reorient_nii(orig_nii, targ_aff="LPS"):
    """
    Reorient to the standard LPS+ DICOM coord.
    """
    if "".join(nib.aff2axcodes(orig_nii.affine)) == targ_aff:
        return orig_nii
    orig_ornt = nib.io_orientation(orig_nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(targ_aff)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    img_orient = orig_nii.as_reoriented(transform)
    return img_orient