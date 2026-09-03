import pickle
import numpy as np
import os
from pathlib import Path
import pandas as pd
import nibabel as nib

base_path = Path('/scratch/hb-EGRET-AAA/projects/UMCG/derivatives/MCMC_CF_nordic')
fs_base_path = Path('/scratch/hb-EGRET-AAA/projects/UMCG/derivatives/freesurfer')

subjects = [f'sub-{i:02d}' for i in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                       19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]]
tasks = ['RET', 'RS1', 'RS2', 'RET2']
hemispheres = ['lh', 'rh']


def load_pkl_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def process_subject_task(subject, task):
    """Merge lh and rh pkl files for one subject/task into a flat list of records."""
    results_path = base_path / subject / 'results'
    merged_data = []

    for hemi in hemispheres:
        filename = f'{hemi}_MCMC_CF_{task}_manual_ses-02.pkl'
        filepath = results_path / filename

        data = load_pkl_file(filepath)
        if data is None:
            print(f"  missing {filename}, skipping")
            continue

        num_voxels = len(data['source_vertex_idx'])
        print(f"  {hemi}: {num_voxels} vertices")

        for i in range(num_voxels):
            # center_idx_chain holds submesh indices across MCMC iterations
            center_chain = data['center_idx_chain'][i]
            center_uncertainty = float(np.std(center_chain))

            ecc_mean_best = float(data['ecc_mean'][i])
            polar_mean_best = float(data['polar_mean'][i])

            # ecc/polar std may not be stored in older pkl files; fall back to center uncertainty
            ecc_uncertainty = float(data['ecc_std'][i]) if hasattr(data, 'ecc_std') else 0.0
            polar_uncertainty = float(data['polar_std'][i]) if hasattr(data, 'polar_std') else 0.0

            if ecc_uncertainty == 0.0 and center_uncertainty > 0:
                ecc_uncertainty = center_uncertainty / 1000.0
            if polar_uncertainty == 0.0 and center_uncertainty > 0:
                polar_uncertainty = center_uncertainty / 1000.0

            source_idx = int(data['source_vertex_idx'][i])
            target_idx = int(data['target_vertex_idx'][i])

            record = {'Subject': subject, 'Task': task, 'Hemisphere': hemi, 'Target_Vertex': target_idx, 'Target_ROI': float(data['target_roi_label'][i]), 'CF_center': source_idx, 'CF_center_uncertainty': center_uncertainty, 'CF_size': float(data['sigma_mean'][i]), 'CF_size_uncertainty': float(data['sigma_std'][i]), 'CF_eccentricity': ecc_mean_best, 'CF_eccentricity_uncertainty': ecc_uncertainty, 'CF_polar_angle': polar_mean_best, 'CF_polar_angle_uncertainty': polar_uncertainty, 'CF_beta': float(data['beta_mean'][i]), 'CF_beta_uncertainty': float(data['beta_std'][i]), 'CF_variance_explained': float(data['r2'][i])}
            merged_data.append(record)

    return merged_data


def main():
    all_records = []

    for subject in subjects:
        subject_path = base_path / subject / 'results'
        if not subject_path.exists():
            print(f"{subject} not found, skipping")
            continue

        print(f"Processing {subject}...")

        for task in tasks:
            print(f"  task {task}...")
            task_data = process_subject_task(subject, task)
            if task_data:
                print(f"    {len(task_data)} records")
                all_records.extend(task_data)
            else:
                print(f"    no data for {task}")

    if not all_records:
        print("ERROR: no records extracted")
        return

    df = pd.DataFrame(all_records)
    output_file = base_path / 'MCMC_CF_merged_manual_results_test_nordic.csv'
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")
    print(f"Output: {output_file}  shape: {df.shape}")


if __name__ == '__main__':
    main()
