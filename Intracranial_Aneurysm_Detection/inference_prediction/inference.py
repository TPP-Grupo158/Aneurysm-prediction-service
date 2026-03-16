import nibabel as nib
from models_classes import reorient_nii, correct_orientation
from prediction_config import DEVICE, LABEL_COLS, CLS_2D_PREDICTOR
from model_initilizer import get_largest_series_files,predict_aneurysm, get_spacing_by_shape, group_dicom_files
import os
import polars as pl
import numpy as np
import pydicom
import dicom2nifti
import shutil
import torch
import gc

def process_single_timepoint(orig_nii, time_index=None):
    """
    Process data for a single timepoint.
    """
    # Extract data for the specified timepoint
    if time_index is not None and orig_nii.ndim == 4 and orig_nii.shape[3] > 1:
        orig_data = orig_nii.get_fdata()[:, :, :, time_index]
        orig_nii = nib.Nifti1Image(orig_data, orig_nii.affine, orig_nii.header)
    
    # Reorient to standard space
    img_orient = reorient_nii(orig_nii, targ_aff="LPS")
    input_img_np = img_orient.get_fdata()
    
    input_img_np = input_img_np.transpose(2, 1, 0)[None]
    original_spacing = img_orient.header.get_zooms()[:3]
    return predict_aneurysm(input_img_np, original_spacing, DEVICE, tta_batch_size=2)

# You can return either a Pandas or Polars dataframe, though Polars is recommended.
def predict(series_path: str) -> pl.DataFrame:
    """
    Make a prediction for a given DICOM series path.
    This function consolidates the core prediction logic into the required format.
    """
    series_id = os.path.basename(series_path)
    
    # Step 1: Quickly retrieve file list
    dicom_files = []
    for root, _, files in os.walk(series_path):
        for file in files:
            # Simplified file type checking
            if file.endswith(('.dcm', '.DCM')) or ('.' not in file and len(file) > 1):
                dicom_files.append(os.path.join(root, file))
    
    if len(dicom_files) == 0:
        probs = np.ones(len(LABEL_COLS)) * 0.5

    elif len(dicom_files) == 1:
        
        ds = pydicom.dcmread(dicom_files[0], force=True)
        
        # Get pixel array
        pixel_array = ds.pixel_array
        # Pixel array shape: (150, 528, 528)
        # Computed spacing: [0.55, 0.5, 0.5]
        
        input_img_np = pixel_array[None]
        spacing = get_spacing_by_shape(pixel_array.shape)
        original_spacing = spacing[::-1]

        print(f"Computed spacing: {spacing}")

        ''' 
        Add exception handling
        '''
        # Filter thick-slice T2 data
        if spacing[0] >= 3:
            print('Detected thick-slice T2 data, performing orientation analysis...')
            
            # ========== Inference from 2D numpy array ==========
            print("\nInference from 2D numpy array")     
            img_3d = pixel_array  # Directly use the loaded pixel_array
            D, _ , _ = pixel_array.shape
            slice_idx = D // 2
            slice_2d = img_3d[slice_idx, :, :]
            
            plane_label, plane_id, plane_conf, plane_probs = CLS_2D_PREDICTOR.inference_from_slice(
                slice_2d=slice_2d)
            
            # Exception handling correction
            if len(pixel_array.shape) == 3 and plane_id != 0:
                print('Entering exception handling...')
                pixel_array, spacing = correct_orientation(pixel_array, spacing, plane_id)
                print(f"Corrected pixel array shape: {pixel_array.shape}")
                print(f"Corrected spacing: {spacing}")
                input_img_np = pixel_array[None]
                original_spacing = spacing[::-1]
        print(input_img_np)
        probs = predict_aneurysm(input_img_np, original_spacing, DEVICE, tta_batch_size=2)
        
    else:
        try: 
            orig_nii = dicom2nifti.dicom_series_to_nifti(series_path, None, reorient_nifti=False)['NII']
        except:
            all_series = group_dicom_files(series_path)
    
            # Find the series with the most slices and sort
            largest_series_files = get_largest_series_files(all_series)
    
            # Now largest_series_files contains the required DICOM file paths sorted by InstanceNumber
            if largest_series_files:
                print(f"\nFinal target series file count: {len(largest_series_files)}")
                # You can now use this list for subsequent image loading and processing
    
            largest_series_path = "/kaggle/largest_series_tmp_path"
    
            if os.path.exists(largest_series_path):
                shutil.rmtree(largest_series_path)
            os.makedirs(largest_series_path)
    
            # 4. Copy files
            for file_path in largest_series_files:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(largest_series_path, filename)
                shutil.copy2(file_path, dest_path)
    
            orig_nii = dicom2nifti.dicom_series_to_nifti(largest_series_path, None, reorient_nifti=False)['NII']
            
        # Process multi-timepoint data
        if orig_nii.ndim == 4 and orig_nii.shape[3] > 1:
            # Get number of timepoints
            t = orig_nii.shape[3]
            
            # Store prediction probabilities for each timepoint
            all_timepoint_probs = []
            
            # Perform inference for each timepoint
            for t_i in range(t):
                print(f"Processing timepoint {t_i + 1}/{t}")
                
                # Use the reused processing function, passing time index
                timepoint_probs = process_single_timepoint(
                    orig_nii,
                    time_index=t_i
                )
                all_timepoint_probs.append(timepoint_probs)
            
            # Combine probabilities from all timepoints, take maximum
            all_timepoint_probs = np.array(all_timepoint_probs)  # shape: (T, prob_length)
            probs = np.max(all_timepoint_probs, axis=0)
            
        else:
            # Single timepoint processing
            probs = process_single_timepoint(
                orig_nii
            )

    pred_df = pl.DataFrame(
        data=[probs.tolist()],
        schema=LABEL_COLS,
        orient='row'
    )
        
    # Perform memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ----------------------------- IMPORTANT ------------------------------
    # You MUST have the following code in your `predict` function
    # to prevent "out of disk space" errors. This is a temporary workaround
    # as we implement improvements to our evaluation system.
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    # ----------------------------------------------------------------------
    
    return pred_df