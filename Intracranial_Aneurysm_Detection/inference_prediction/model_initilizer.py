from prediction_config import COMPILE_NETWORK, MODEL_PATHS, USE_NUM_GPUS, get_device, DEVICE, executor,GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR
from nnXNet.nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import nnXNetPredictor
from nnXNet.nnxnet.inference.predict_from_raw_data_two_seg_with_cls_no_seg_return_no_filter import nnXNetPredictor as nnXNetPredictorWithCls
from nnXNet.nnxnet.utilities.helpers import empty_cache, dummy_context
from concurrent.futures import  as_completed
import torch
from models_classes import PlaneClassifier
import pydicom
import os
from typing import Tuple,  List, Tuple, Dict, Any
import numpy as np

def may_compile_network(network):
    if COMPILE_NETWORK:
        return torch.compile(network)
    return network

def init_predictors(device):
    global GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR
    
    # If already initialized, return the global instance directly
    if GLOBAL_VESSEL_ROI_PREDICTOR is not None and GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS is not None and CLS_2D_PREDICTOR is not None:
        return GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR
    
    # Stage 1: Vessel ROI Segmentation (Using the new multiplanar predictor)
    GLOBAL_VESSEL_ROI_PREDICTOR = nnXNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=device,
        allow_tqdm=False
    )
    GLOBAL_VESSEL_ROI_PREDICTOR.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_PATHS['vessel_ROI_seg'],
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    GLOBAL_VESSEL_ROI_PREDICTOR.initialize_network_and_gaussian()
    GLOBAL_VESSEL_ROI_PREDICTOR.network = may_compile_network(GLOBAL_VESSEL_ROI_PREDICTOR.network)

    # Stage2: Aneurysm classification
    aneurysm_predictor_f0 = nnXNetPredictorWithCls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=get_device(gpu_id=0),
        verbose=False
    )

    aneurysm_predictor_f1 = nnXNetPredictorWithCls(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=get_device(gpu_id=0),
        verbose=False
    )

    aneurysm_predictor_f0.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_PATHS['aneurysm_cls_1'],
        use_folds=(0, ),
        checkpoint_name="checkpoint_final.pth",
    )
    aneurysm_predictor_f0.initialize_network_and_gaussian()

    aneurysm_predictor_f1.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_PATHS['aneurysm_cls_2'],
        use_folds=(1, ),
        checkpoint_name="checkpoint_final.pth",
    )
    aneurysm_predictor_f1.initialize_network_and_gaussian()

    aneurysm_predictor_f0.network = may_compile_network(aneurysm_predictor_f0.network)
    aneurysm_predictor_f1.network = may_compile_network(aneurysm_predictor_f1.network)

    # ========== Initialize 2D Orientation Classifier Inferencer ==========
    CLS_2D_PREDICTOR = PlaneClassifier(
        checkpoint_path=MODEL_PATHS['plane_2d_cls'],
        device=device,
        target_size=(256, 256)
    )
    CLS_2D_PREDICTOR.model = may_compile_network(CLS_2D_PREDICTOR.model)

    GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS = [aneurysm_predictor_f0, aneurysm_predictor_f1]
    return GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS, CLS_2D_PREDICTOR

    
GLOBAL_VESSEL_ROI_PREDICTOR, GLOBAL_ANEURYSM_PREDICTOR, CLS_2D_PREDICTOR = init_predictors(DEVICE)

def group_dicom_files(study_folder_path: str) -> Dict[Tuple, List[str]]:
    """
    Groups DICOM files into pseudo-series based on StudyInstanceUID, 
    FrameOfReferenceUID, Modality, and ImageOrientationPatient.
    """
    dicom_groups = {}
    dicom_files = []

    for root, _, files in os.walk(study_folder_path):
        for file in files:
            if file.endswith(('.dcm', '.DCM')) or ('.' not in file and len(file) > 1):
                dicom_files.append(os.path.join(root, file))

    if not dicom_files:
        print(f"No DICOM files found in path: {study_folder_path}")
        return dicom_groups
    
    print(f"Found {len(dicom_files)} files in total, starting grouping...")

    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            study_uid = getattr(ds, 'StudyInstanceUID', 'NO_STUDY_UID')
            frame_uid = getattr(ds, 'FrameOfReferenceUID', 'NO_FRAME_UID')
            modality = getattr(ds, 'Modality', 'UNKNOWN')
            
            orientation = getattr(ds, 'ImageOrientationPatient', [0, 0, 0, 0, 0, 0])
            orientation_key = tuple(np.round(orientation, 4)) 

            group_key = (study_uid, frame_uid, modality, orientation_key)

            if group_key not in dicom_groups:
                dicom_groups[group_key] = []
            dicom_groups[group_key].append(file_path)

        except (pydicom.errors.InvalidDicomError, Exception):
            continue

    print("-" * 50)
    print(f"Grouping completed. Identified {len(dicom_groups)} logical series in total.")
    return dicom_groups

def get_largest_series_files(all_series: Dict[Tuple, List[str]]) -> List[str]:
    """
    Finds the series with the most slices and returns the file list sorted by InstanceNumber.

    Args:
        all_series (dict): Grouping dictionary returned by group_dicom_files.

    Returns:
        list: List of DICOM file paths from the series with the most slices, sorted by InstanceNumber.
    """
    if not all_series:
        return []

    # 1. Find the series with the most slices
    # (Key, file list)
    max_layers_series_item = None
    max_layers = 0

    for group_key, file_list in all_series.items():
        if len(file_list) > max_layers:
            max_layers = len(file_list)
            max_layers_series_item = (group_key, file_list)

    if not max_layers_series_item:
        print("No valid series found.")
        return []

    target_group_key, target_series_files = max_layers_series_item
    
    print("-" * 50)
    print(f"*** Found series with the most slices *** (Slice count: {max_layers})")
    print(f"  Modality: {target_group_key[2]}")
    print(f"  Orientation: {target_group_key[3][:3]}...")

    # 2. Sort the target series (using InstanceNumber)
    sorted_files_with_number: List[Tuple[Any, str]] = []
    
    # Iterate through files to get InstanceNumber
    for fp in target_series_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            # Sort by InstanceNumber. Use 0 if InstanceNumber doesn't exist.
            # Alternatively, consider using ImagePositionPatient[2] for sorting
            instance_number = getattr(ds, 'InstanceNumber', 0)
            sorted_files_with_number.append((instance_number, fp))
        except:
            pass

    # Sort by InstanceNumber
    sorted_files_with_number.sort(key=lambda x: x[0])
    
    final_file_paths = [fp for _, fp in sorted_files_with_number]
    
    return final_file_paths

def get_spacing_by_shape(shape):
    """
    Efficiently maps spacing based on the size of each axis.
    Rules:
    - Axis > 300: 0.5mm
    - 120 < Axis <= 300: 0.55mm  
    - 100 < Axis <= 120: 0.75mm
    - 80 < Axis <= 100: 1.0mm
    - 60 < Axis <= 80: 1.5mm
    - 45 < Axis <= 60: 3.0mm
    - Axis <= 45: 5.0mm
    """
    spacing = []
    for dim in shape:
        if dim > 300:
            spacing.append(0.5)
        elif dim > 120:
            spacing.append(0.55)
        elif dim > 100:
            spacing.append(0.75)
        elif dim > 80:
            spacing.append(1.0)
        elif dim > 60:
            spacing.append(1.5)
        elif dim > 45:
            spacing.append(3.0)
        else:
            spacing.append(5.0)
    return spacing

def flip_z(img_tensor: torch.Tensor) -> torch.Tensor:
    """Flip along Z-axis (superior-inferior): corresponds to dim=2"""
    # Shape [B, C, D, H, W], D corresponds to dim=2
    return torch.flip(img_tensor, dims=[2])

def flip_y(img_tensor: torch.Tensor) -> torch.Tensor:
    """Flip along Y-axis (anterior-posterior): corresponds to dim=3"""
    # Shape [B, C, D, H, W], H corresponds to dim=3
    return torch.flip(img_tensor, dims=[3])

def flip_x(tensor):
    """Flip along X-axis (left-right): corresponds to dim=4"""
    # Shape [B, C, D, H, W], W corresponds to dim=4
    return torch.flip(tensor, dims=[4])

@torch.no_grad()
def worker_infer(num_cls_task, tta_batch_size, image_resized, aneurysm_predictor_fold, fold_i, all_fold_mean_logits_by_task):
    """
    image_resized: can be on whatever device, because `image_resized = image_resized.to(device)` 
        will move it to correct device for inference. 
    Results are on CPU.
    all_fold_mean_logits_by_task: modified inplace in this function.
        It is thread-safe if we make sure different threads write to different slots (or sub-slots) of the list.
    """
    device = aneurysm_predictor_fold.device
    print(f"[*] Worker Using Device: {device}")

    # Task 2 (Location Classification) Left-right flip index mapping (13 classes: 0-12)
    # [L_ICL, R_ICL, L_SCL, R_SCL, L_MCA, R_MCA, AC, L_AC, R_AC, L_PC, R_PC, BT, OP]
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    # Swap: 1, 0, 3, 2, 5, 4, 6, 8, 7, 10, 9, 11, 12
    # Only for Task 2 logits (shape [N_classes=13])
    TASK2_SWAP_INDICES = [1, 0, 3, 2, 5, 4, 6, 8, 7, 10, 9, 11, 12]

    image_resized = image_resized.to(device)
    # ======================================================
    # TTA Step 1: Construct all TTA augmented image list
    # Add X-axis flip (left-right flip)
    # ======================================================
    
    # 1. Original image
    I_orig = image_resized
    # 2. Augmentation A: Y-axis flip (anterior-posterior)
    I_flip_y = flip_y(I_orig)
    # 3. Augmentation B: Z-axis flip (superior-inferior)
    I_flip_z = flip_z(I_orig)
    # 4. Augmentation C: Y-axis + Z-axis flip
    I_flip_yz = flip_y(I_flip_z)
    
    # 5. Augmentation D: X-axis flip (left-right)
    I_flip_x = flip_x(I_orig)
    # 6. Augmentation E: X-axis + Y-axis flip
    I_flip_xy = flip_x(I_flip_y)
    # 7. Augmentation F: X-axis + Z-axis flip
    I_flip_xz = flip_x(I_flip_z)
    # 8. Augmentation G: X-axis + Y-axis + Z-axis flip
    I_flip_xyz = flip_x(I_flip_yz)
    
    # TTA list (total 8 augmentations)
    tta_images = [
        I_orig, I_flip_y, I_flip_z, I_flip_yz,
        I_flip_x, I_flip_xy, I_flip_xz, I_flip_xyz
    ]
    
    # Mark which augmentations performed X-axis (left-right) flip
    # 0: not flipped, 1: X-axis flipped
    # Corresponds to tta_images list
    x_flip_masks = [0, 0, 0, 0, 1, 1, 1, 1] 

    # ======================================================
    # TTA Step 2: Iterative inference (supports tta_batch_size setting)
    # ======================================================
    
    num_tta_augments = len(tta_images) # 8
    # current_fold_tta_logits_by_task[task_i] is a list storing all TTA image logits for that task
    current_fold_tta_logits_by_task = [[] for _ in range(num_cls_task)]
    
    empty_cache(device)
    
    for i in range(0, num_tta_augments, tta_batch_size):
        # Construct current batch of TTA images and corresponding flip flags
        batch_tta_images = tta_images[i:i + tta_batch_size]
        batch_x_flip_masks = x_flip_masks[i:i + tta_batch_size]
        
        # Stack into TTA Batch (shape: [tta_batch_size, 1, 224, 224, 224])
        image_tta_batch = torch.cat(batch_tta_images, dim=0)

        with torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            # predicted_logits_list shape: [task1_logits, task2_logits],
            predicted_logits_list = aneurysm_predictor_fold.network(image_tta_batch, only_forward_cls=True)
            
            # Collect logits for each task
            # Task 1 (first task, index 0): No left-right swap (assuming Aneurysm Present)
            current_fold_tta_logits_by_task[0].append(predicted_logits_list[0])
            
            # Task 2 (second task, index 1): Handle left-right swap
            current_task2_logits = predicted_logits_list[1]
            
            processed_logits_list = []
            for j, is_x_flipped in enumerate(batch_x_flip_masks):
                # If current TTA image performed X-axis flip (left-right flip)
                if is_x_flipped == 1:
                    # Swap left-right positions for Task 2 results
                    flipped_logits = current_task2_logits[j][TASK2_SWAP_INDICES]
                    processed_logits_list.append(flipped_logits)
                else:
                    # Otherwise keep original logits
                    processed_logits_list.append(current_task2_logits[j])
                    
            # Collect processed Task 2 logits
            current_fold_tta_logits_by_task[1].append(torch.stack(processed_logits_list))
        
        del image_tta_batch
        empty_cache(device)


    # --------------------------------------------------
    # TTA Step 3: Current Fold result integration (Logits averaging)
    # --------------------------------------------------
    for task_i in range(num_cls_task):
        # Merge all batch logits for current fold into a large [num_tta_augments, num_classes] Tensor
        logits_tta = torch.cat(current_fold_tta_logits_by_task[task_i], dim=0)
        
        # Logits averaging (along dimension 0, i.e., batch dimension)
        mean_logit_fold = logits_tta.mean(dim=0, keepdim=False).cpu() # shape: [num_classes]
        
        # Collect current fold's TTA average logits
        all_fold_mean_logits_by_task[task_i][fold_i] = mean_logit_fold
    

def predict_aneurysm(input_img_np, original_spacing, device, tta_batch_size=2):
    """
    Aneurysm prediction function (with left-right flip TTA and left-right swapping for Task 2 results)
    
    Args:
        input_img_np: Input image numpy array
        original_spacing: Original image spacing (x, y, z)
        device: Computing device (cpu/cuda)
    
    Returns:
        timepoint_probs: Classification probability array [task2 probabilities, task1 probabilities]
    """
    # Stage 1: Vessel ROI prediction
    stage_1_target_spacing = np.array([1, 0.55, 0.5])
    with torch.no_grad():
        z_min_final, z_max_final, y_min_final, y_max_final, x_min_final, x_max_final = GLOBAL_VESSEL_ROI_PREDICTOR.predict_from_multi_axial_slices(
            input_img_np, original_spacing, stage_1_target_spacing, max_batch_size=16
        )

    # Crop image to ROI region
    img_cropped_np = input_img_np[0][z_min_final:z_max_final, y_min_final:y_max_final, x_min_final:x_max_final][None]

    del input_img_np 

    img_cropped_np = np.ascontiguousarray(img_cropped_np)

    # Stage 2: Aneurysm classification
    img_cropped_tensor = torch.from_numpy(img_cropped_np).half().to(device)  # HOUJING: added half()

    # Image normalization
    image_normed = (img_cropped_tensor - img_cropped_tensor.mean()) / img_cropped_tensor.std().clip(1e-8)

    del img_cropped_tensor

    dst_shape = [224, 224, 224]
    
    # Image resampling
    image_resized = torch.nn.functional.interpolate(
        image_normed[None], size=dst_shape, mode='trilinear', align_corners=True
    )

    del image_normed

    # Initialize a list to store logits for each TTA image
    # all_fold_mean_logits_by_task[task_i] is a list storing all TTA image logits for that task
    num_cls_task = 2
    n_folds = 2
    all_fold_mean_logits_by_task = [[None for _ in range(n_folds)] for _ in range(num_cls_task)]

     # ======================================================
    # Inference for each fold model in parallel
    # ======================================================
    futures = []
    for fold_i, aneurysm_predictor_fold in enumerate(GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS):
        # This runs asynchronously
        futures.append(executor.submit(worker_infer, num_cls_task, tta_batch_size, image_resized, aneurysm_predictor_fold, fold_i, all_fold_mean_logits_by_task))
    # Iterating over as_completed(futures) ensures all tasks finish before moving on
    for future in as_completed(futures):
        pass
        
    # ======================================================
    # Final result integration (averaging TTA mean logits across all folds)
    # ======================================================
    
    aggregated_probs_list = []
    
    for task_i in range(num_cls_task):
        logits_tta = torch.stack(all_fold_mean_logits_by_task[task_i], dim=0)
        
        mean_logit = logits_tta.mean(dim=0, keepdim=False)
        
        # Convert to final probabilities using Sigmoid
        final_prob = torch.sigmoid(mean_logit)
        
        aggregated_probs_list.append(final_prob.to('cpu').numpy().flatten())

    # Merge probabilities: Task2 first, Task1 last (consistent with original code)
    task1_probs = aggregated_probs_list[0] # Task 1 (existence)
    task2_probs = aggregated_probs_list[1] # Task 2 (location)

    return np.concatenate([task2_probs, task1_probs], axis=0)