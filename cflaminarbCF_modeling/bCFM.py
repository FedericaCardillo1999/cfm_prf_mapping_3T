# %% [markdown]
# # <span style="color:#065C9C">**Benchmarks** for different CF modelling strategies </span>

# ### CF center restriction and Gaussian weighting update (2026-05-04)
# CF center candidates are now limited to V1 vertices within the atlas eccentricity cap
# (<=20 deg for Benson, <=7 deg for manual), while the CF stimulus uses the full V1 source
# timecourses so the Gaussian can weight neighboring V1 data beyond the restricted center set.

# %%
import argparse, os
import shutil
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description="CF modeling pipeline")
parser.add_argument("project", choices=["UMCG", "OVGU", "7T"])
parser.add_argument("hemisphere", choices=["lh", "rh"])
parser.add_argument("task", choices=["RET", "RET2", "RS1", "RS2", "RS"])
parser.add_argument("subject")
parser.add_argument("session")

args = parser.parse_args()

project = args.project
h = args.hemisphere
task = args.task
subject_id = args.subject

if project == "UMCG":
    session_id = "02"
else:
    if args.session is None:
        parser.error("Session ID required for OVGU (use --session)")
    session_id = args.session

all_results = {}

fs_subject_path = f"/scratch/hb-EGRET-AAA/projects/{project}/derivatives/freesurfer/sub-{subject_id}"
root_dir = os.path.join(
    f"/scratch/hb-EGRET-AAA/projects/{project}/derivatives",
    "MCMC_CF_nordic",
    f"sub-{subject_id}"
)
fig_dir = os.path.join(root_dir, "figures")
results_dir = os.path.join(root_dir, "results")
surf_dir = os.path.join(root_dir, "surf")
for d in (fig_dir, results_dir, surf_dir):
    os.makedirs(d, exist_ok=True)

# copy inflated surfaces locally for visualization
for hemi in ['lh', 'rh']:
    src_inflated = os.path.join(fs_subject_path, 'surf', f'{hemi}.inflated')
    dst_inflated = os.path.join(surf_dir, f'{hemi}.inflated')
    if os.path.exists(src_inflated):
        try:
            shutil.copy2(src_inflated, dst_inflated)
        except Exception as e:
            print(f'warning: could not copy {src_inflated}: {e}')
    else:
        print(f'warning: inflated surface not found: {src_inflated}')
suffix = ""

if project in ["7T", "OVGU"]:
    atlas_list = ["benson"]
else:
    atlas_list = ["manual", "benson"]

for atlas in atlas_list:

    print(f"\nRunning atlas: {atlas}")

    if atlas == "benson":
        ecc_range_target = (0, 20.0)
        ecc_range_source = (0, 20.0)
    else:
        ecc_range_target = (0, 7.0)
        ecc_range_source = (0, 7.0)

    ecc_range = ecc_range_target

    # load empirical pRF results only for manual atlas
    if atlas == "manual":
        task_for_prf = task if task.startswith("RET") else "RET"
        prf_pkl_path = f"/scratch/hb-EGRET-AAA/projects/{project}/derivatives/pRFM/sub-{subject_id}/ses-{session_id}/nordic/model-{atlas}-nelder-mead-GM_desc-prf_params_{task_for_prf}.pkl"
        try:
            with open(prf_pkl_path, "rb") as f:
                prf = pickle.load(f)
        except FileNotFoundError:
            print(f"pRF file not found at {prf_pkl_path}")
            prf = None
        except Exception as e:
            print(f"error loading pRF: {e}")
            prf = None
    else:
        prf = None

    ts_dir = f"/scratch/hb-EGRET-AAA/projects/{project}/derivatives/pRFM"
    task_name = "RestingState" if task.startswith("RS") else task

    import os
    import sys
    import pickle

    superficial_depth = "0.7777777777777778"
    middle_depth = "0.5555555555555556"

    if project == "7T":
        if task_name == "RestingState":
            ts_path_source = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/depth-{superficial_depth}/sub-{subject_id}_ses-{session_id}_task-{task_name}_hemi-{h}.npy"
            ts_path_target = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/depth-{middle_depth}/sub-{subject_id}_ses-{session_id}_task-{task_name}_hemi-{h}.npy"
        else:
            ts_path_source = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/depth-{superficial_depth}/sub-{subject_id}_ses-{session_id}_task-{task_name}_hemi-{h}_desc-avg.npy"
            ts_path_target = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/depth-{middle_depth}/sub-{subject_id}_ses-{session_id}_task-{task_name}_hemi-{h}_desc-avg.npy"
    else:
        if task_name == "RestingState":
            run_folder = "run-1" if task in ["RS1", "RS"] else "run-2" if task == "RS2" else "run-1"
            ts_path_source = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/{run_folder}/sub-{subject_id}_ses-{session_id}_task-{task_name}_{run_folder}_hemi-{h}_trimmed_bold_GM.npy"
            ts_path_target = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/{run_folder}/sub-{subject_id}_ses-{session_id}_task-{task_name}_{run_folder}_hemi-{h}_trimmed_bold_GM.npy"
        else:
            ts_path_source = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/trimmed/sub-{subject_id}_ses-{session_id}_task-{task_name}_hemi-{h}_desc-avg_trimmed_bold_GM.npy"
            ts_path_target = f"{ts_dir}/sub-{subject_id}/ses-{session_id}/nordic_sm4/trimmed/sub-{subject_id}_ses-{session_id}_task-{task_name}_hemi-{h}_desc-avg_trimmed_bold_GM.npy"

    import os
    import sys
    import math
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")
    import numpy as np
    import seaborn as sns
    import scipy.stats as stats
    import pandas as pd
    import nibabel as nib
    import neuropythy as ny
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from prfpy.stimulus import CFStimulus
    from prfpy.model import CFGaussianModel
    from prfpy.fit import CFFitter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Wedge, Rectangle
    import matplotlib.colors as mcolors
    from cfmap.color_palettes import get_color_palettes
    import importlib
    import cfmap.pRF_processing
    from cfmap.pRF_processing import plot_prf_histograms, plot_roi_retMap
    importlib.reload(cfmap.pRF_processing)
    from cfmap.CF_mapping import optimize_connfield_dfree, optimize_connfield_joint, plot_convergence_summary_table, compare_cf_results
    from scipy.stats import pearsonr
    from pingouin import circ_corrcl

    # load FreeSurfer subject and compute V1-centred orthographic flatmaps
    sub = ny.freesurfer_subject(fs_subject_path)
    v1_centers = {}
    v1_rights = {}
    hemi_order = [h, 'rh' if h == 'lh' else 'lh']
    for hemi in hemi_order:
        if atlas == "manual":
            label_path = f"{fs_subject_path}/label/{hemi}.V1_manual.label"
            V1_vertices = nib.freesurfer.io.read_label(label_path)
        elif atlas == "benson":
            varea = nib.load(os.path.join(fs_subject_path, "surf", f"{hemi}.benson14_varea.mgz")).get_fdata().squeeze()
            V1_vertices = np.where(varea == 1)[0]
        coords = sub.hemis[hemi].surface().coordinates.T
        coords = coords[V1_vertices]
        coords = coords[np.isfinite(coords).all(axis=1)]
        center = coords.mean(axis=0)
        lateral = coords[np.argmax(coords[:, 0])]
        right_vec = lateral - center
        right_vec = right_vec / np.linalg.norm(right_vec)
        v1_centers[hemi] = center
        v1_rights[hemi] = right_vec

    map_projs = {h: ny.map_projection(chirality=h, center=v1_centers[h], center_right=v1_rights[h],
                                       method='orthographic', radius=np.pi/2, registration='native')
                 for h in ['lh', 'rh']}
    flatmaps = {h: mp(sub.hemis[h]) for h, mp in map_projs.items()}

    # cortex_idx maps each flatmap vertex back to its native FreeSurfer vertex index
    cortex_idx = flatmaps[h].prop('index')
    surf_atlas_path = os.path.join(fs_subject_path, "surf")
    benson_eccen_full = nib.load(os.path.join(surf_atlas_path, f'{h}.benson14_eccen.mgz')).get_fdata().squeeze()
    benson_polar_full = nib.load(os.path.join(surf_atlas_path, f'{h}.benson14_angle.mgz')).get_fdata().squeeze()
    benson_sigma_full = nib.load(os.path.join(surf_atlas_path, f'{h}.benson14_sigma.mgz')).get_fdata().squeeze()
    ecc_atlas   = benson_eccen_full[cortex_idx]
    polar_atlas = benson_polar_full[cortex_idx]
    sigma_atlas = benson_sigma_full[cortex_idx]

    x_atlas = ecc_atlas * np.cos(polar_atlas)
    y_atlas = ecc_atlas * np.sin(polar_atlas)
    prf_results_atlas = pd.DataFrame({'x': x_atlas, 'y': y_atlas, 'sd': sigma_atlas, 'r2': np.ones_like(sigma_atlas)})

    if prf is not None:
        pars = prf['model'].iterative_search_params
        mask = prf['rois_mask'].astype(bool)
    else:
        pars = None
        mask = None

    n_lh = sub.hemis['lh'].vertex_count
    n_rh = sub.hemis['rh'].vertex_count
    n_total = n_lh + n_rh
    x_native     = np.full(n_total, np.nan)
    y_native     = np.full(n_total, np.nan)
    sigma_native = np.full(n_total, np.nan)
    r2_native    = np.full(n_total, np.nan)

    if pars is not None and mask is not None:
        x_native[mask]     = pars[:, 0]
        y_native[mask]     = pars[:, 1]
        sigma_native[mask] = pars[:, 2]
        r2_native[mask]    = pars[:, 7]

    if h == 'lh':
        x_full     = x_native[:n_lh]
        y_full     = y_native[:n_lh]
        sigma_full = sigma_native[:n_lh]
        r2_full    = r2_native[:n_lh]
    else:
        x_full     = x_native[n_lh:]
        y_full     = y_native[n_lh:]
        sigma_full = sigma_native[n_lh:]
        r2_full    = r2_native[n_lh:]

    x_plot     = x_full[cortex_idx]
    y_plot     = y_full[cortex_idx]
    sigma_plot = sigma_full[cortex_idx]
    r2_plot    = r2_full[cortex_idx]

    if pars is not None and mask is not None:
        prf_results_manual = pd.DataFrame({'x': x_plot, 'y': y_plot, 'sd': sigma_plot, 'r2': r2_plot})
    else:
        prf_results_manual = prf_results_atlas.copy()

    color_palettes = get_color_palettes()
    colors_ecc = color_palettes["eccentricity"]
    colors_polar = color_palettes["polar"]

    cortex_idx = flatmaps[h].prop('index')
    if h == 'lh':
        x_hemi = x_native[:n_lh]; y_hemi = y_native[:n_lh]
        sigma_hemi = sigma_native[:n_lh]; r2_hemi = r2_native[:n_lh]
    else:
        x_hemi = x_native[n_lh:]; y_hemi = y_native[n_lh:]
        sigma_hemi = sigma_native[n_lh:]; r2_hemi = r2_native[n_lh:]

    if pars is not None and mask is not None:
        x_manual     = x_hemi[cortex_idx]
        y_manual     = y_hemi[cortex_idx]
        sigma_manual = sigma_hemi[cortex_idx]
        r2_manual    = r2_hemi[cortex_idx]
    else:
        x_manual = x_atlas; y_manual = y_atlas
        sigma_manual = sigma_atlas; r2_manual = np.ones_like(sigma_atlas)

    ecc_manual   = np.abs(x_manual + 1j*y_manual)
    polar_manual = np.angle(x_manual + 1j*y_manual)

    # ROI masks
    flat_idx   = flatmaps[h].prop('index')
    n_vertices = len(flat_idx)

    if atlas == "manual":
        V1_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V1.label")
        V2_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V2.label")
        V3_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V3.label")
        V4_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V4.label")
        LO_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_LO.label")
    elif atlas == "benson":
        varea = nib.load(os.path.join(fs_subject_path, "surf", f"{h}.benson14_varea.mgz")).get_fdata().squeeze()
        V1_vertices  = np.where(varea == 1)[0]
        V2_vertices  = np.where(varea == 2)[0]
        V3_vertices  = np.where(varea == 3)[0]
        V4_vertices  = np.where(varea == 4)[0]
        LO1_vertices = np.where(varea == 7)[0]
        LO2_vertices = np.where(varea == 8)[0]

    V1_mask = np.isin(flat_idx, V1_vertices)
    V2_mask = np.isin(flat_idx, V2_vertices)
    V3_mask = np.isin(flat_idx, V3_vertices)
    V4_mask = np.isin(flat_idx, V4_vertices)
    if atlas == "manual":
        LO_mask = np.isin(flat_idx, LO_vertices)
    else:
        LO1_mask = np.isin(flat_idx, LO1_vertices)
        LO2_mask = np.isin(flat_idx, LO2_vertices)

    # eccentricity mask for target ROIs; V1_mask stays anatomically complete
    if pars is None:
        ecc = ecc_atlas
    else:
        ecc = np.abs(x_plot + 1j*y_plot)
    ecc_mask = (ecc >= ecc_range_target[0]) & (ecc <= ecc_range_target[1])

    # V1_mask_for_viz is the ecc-restricted version used only for plotting/source selection
    V1_final_for_viz = V1_mask & ecc_mask
    V2_final = V2_mask & ecc_mask
    V3_final = V3_mask & ecc_mask
    V4_final = V4_mask & ecc_mask
    if atlas == "manual":
        LO_final = LO_mask & ecc_mask
    else:
        LO1_final = LO1_mask & ecc_mask
        LO2_final = LO2_mask & ecc_mask

    if project == "UMCG" and atlas == "manual":
        target_roi_mask = V2_final | V3_final | V4_mask | LO_mask
    else:
        target_roi_mask = V2_final | V3_final | V4_final | LO1_final | LO2_final
    source_roi_mask = V1_final_for_viz

    # ROI plot
    plot_colors = np.zeros((n_vertices, 4))
    plot_colors[V1_final_for_viz] = [1, 0, 0, 1]
    plot_colors[V2_final] = plt.matplotlib.colors.to_rgba("#1f77b4")
    if project == "UMCG" and atlas == "manual":
        plot_colors[V3_final] = plt.matplotlib.colors.to_rgba("#ff7f0e")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ny.cortex_plot(flatmaps[h], axes=ax, color=plot_colors)
    legend_elements = [
        Patch(facecolor='red', label='V1'),
        Patch(facecolor="#1f77b4", label='V2'),
        Patch(facecolor="#ff7f0e", label='V3'),
        Patch(facecolor="#2ca02c", label='V4'),
    ]
    if atlas == "manual":
        legend_elements.append(Patch(facecolor="#d62728", label='LO'))
    else:
        legend_elements.append(Patch(facecolor="#d62728", label='LO1'))
        legend_elements.append(Patch(facecolor="#9467bd", label='LO2'))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="black")
    ax.set_aspect('equal')
    ax.set_title(f"Source (V1) and Target (V2/V3/V4/LO) ROIs - {atlas}", fontsize=11)
    ax.axis('off')
    fig.savefig(os.path.join(fig_dir, f"{h}_ROIs_{atlas}_{task}_ses-{session_id}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # compare empirical vs atlas retinotopy within V1
    os.makedirs(fig_dir, exist_ok=True)
    ecc   = np.abs(x_plot + 1j*y_plot)
    polar = np.angle(x_plot + 1j*y_plot)
    r2    = r2_plot
    ecc_v1       = np.where(V1_mask, ecc,       np.nan)
    polar_v1     = np.where(V1_mask, polar,     np.nan)
    r2_v1        = np.where(V1_mask, r2,        np.nan)
    ecc_atlas_v1 = np.where(V1_mask, ecc_atlas, np.nan)
    polar_atlas_v1 = np.where(V1_mask, polar_atlas, np.nan)

    # build submesh from the FULL V1 mask (no ecc restriction) so distance matrix covers all V1
    roi_submesh = flatmaps[h].submesh(V1_mask)

    _submesh_native  = np.asarray(roi_submesh.labels)
    _flat_idx_arr    = np.asarray(flat_idx)
    _native_to_flat  = np.full(_flat_idx_arr.max() + 1, -1, dtype=np.intp)
    _native_to_flat[_flat_idx_arr] = np.arange(len(_flat_idx_arr))
    _submesh_flat_pos = _native_to_flat[_submesh_native]

    n_v1_full = int(np.sum(V1_mask))
    if len(_submesh_flat_pos) != n_v1_full:
        print(f"  note: {n_v1_full - len(_submesh_flat_pos)} isolated V1 vertices dropped by submesh")

    # V1_mask_submesh is the authoritative source mask going forward
    V1_mask_submesh = np.zeros(len(flat_idx), dtype=bool)
    V1_mask_submesh[_submesh_flat_pos] = True
    submesh_flat_positions = np.where(V1_mask_submesh)[0]
    submesh_indices = submesh_flat_positions

    ecc_pRF       = ecc[submesh_indices]
    pol_pRF       = polar[submesh_indices]
    ecc_atlas_vals = ecc_atlas[submesh_indices]
    pol_atlas_vals = polar_atlas[submesh_indices]
    valid_ecc = np.isfinite(ecc_pRF) & np.isfinite(ecc_atlas_vals)
    valid_pol = np.isfinite(pol_pRF) & np.isfinite(pol_atlas_vals)

    distance_matrix = roi_submesh.dijkstra(0, 1)[0]

    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(distance_matrix, cmap='viridis', aspect='equal')
    plt.colorbar(label='Cortical distance (mm)', shrink=0.8)
    plt.title('V1 cortical distance matrix')
    plt.xlabel('Vertex i'); plt.ylabel('Vertex j')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{h}_distance_matrix_{atlas}_{task}_ses-{session_id}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # CF model fitting
    if atlas == "manual":
        V1_vertices = nib.freesurfer.io.read_label(os.path.join(fs_subject_path, "label", f"{h}.manual_V1.label"))
        V2_vertices = nib.freesurfer.io.read_label(os.path.join(fs_subject_path, "label", f"{h}.manual_V2.label"))
        V3_vertices = nib.freesurfer.io.read_label(os.path.join(fs_subject_path, "label", f"{h}.manual_V3.label"))
        V4_vertices = nib.freesurfer.io.read_label(os.path.join(fs_subject_path, "label", f"{h}.manual_V4.label"))
        LO_vertices = nib.freesurfer.io.read_label(os.path.join(fs_subject_path, "label", f"{h}.manual_LO.label"))
    elif atlas == "benson":
        varea = nib.load(os.path.join(fs_subject_path, "surf", f"{h}.benson14_varea.mgz")).get_fdata().squeeze()
        V1_vertices  = np.where(varea == 1)[0]
        V2_vertices  = np.where(varea == 2)[0]
        V3_vertices  = np.where(varea == 3)[0]
        V4_vertices  = np.where(varea == 4)[0]
        LO1_vertices = np.where(varea == 7)[0]
        LO2_vertices = np.where(varea == 8)[0]

    if project == "UMCG" and atlas == "manual":
        target_roi_mask = V2_final | V3_final | V4_mask | LO_mask
    else:
        target_roi_mask = V2_final | V3_final | V4_final | LO1_final | LO2_final

    ts_source = np.nan_to_num(np.load(ts_path_source))
    ts_target = np.nan_to_num(np.load(ts_path_target))

    ts_source_flat = ts_source[:, flat_idx]
    ts_target_flat = ts_target[:, flat_idx]

    # source data uses V1_mask_submesh (submesh-synchronised) to guarantee correct sizing
    sub_idx     = np.where(V1_mask_submesh)[0]
    source_data = ts_source_flat[:, sub_idx]
    target_data = ts_target_flat[:, target_roi_mask]

    if target_data.shape[1] != np.sum(target_roi_mask):
        print(f"WARNING: target_data size mismatch ({target_data.shape[1]} vs {np.sum(target_roi_mask)})")

    distance_matrix = roi_submesh.dijkstra(0, 1)[0]

    source_stim = CFStimulus(source_data.T, np.arange(source_data.shape[1]), distance_matrix)
    model = CFGaussianModel(source_stim)
    gf = CFFitter(data=target_data.T, model=model)
    sigmas = np.linspace(0.5, 15, 20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gf.grid_fit(sigmas, verbose=False, n_batches=100)

    r2_threshold = 0.1
    optimized_results = optimize_connfield_dfree(gf, r2_threshold=r2_threshold, sigma_bounds=[(0.1, 20.0)], method='L-BFGS-B')

    gradescent_results = None

    distance_matrix = roi_submesh.dijkstra(0, 1)[0]

    # index maps: submesh → flatmap and flatmap → target
    submesh_flat_indices = np.where(V1_mask_submesh)[0]
    target_indices = np.where(target_roi_mask)[0]

    # target ROI flatmap colors for visualization
    n_vertices = len(flat_idx)
    if atlas == "manual":
        V2_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V2.label")
        V3_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V3.label")
        V4_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_V4.label")
        LO_vertices = nib.freesurfer.io.read_label(f"{fs_subject_path}/label/{h}.manual_LO.label")
    elif atlas == "benson":
        varea = nib.load(os.path.join(fs_subject_path, "surf", f"{h}.benson14_varea.mgz")).get_fdata().squeeze()
        V2_vertices  = np.where(varea == 2)[0]
        V3_vertices  = np.where(varea == 3)[0]
        V4_vertices  = np.where(varea == 4)[0]
        LO1_vertices = np.where(varea == 7)[0]
        LO2_vertices = np.where(varea == 8)[0]

    V2_mask = np.isin(flat_idx, V2_vertices)
    V3_mask = np.isin(flat_idx, V3_vertices)
    V4_mask = np.isin(flat_idx, V4_vertices)
    if atlas == "manual":
        LO_mask = np.isin(flat_idx, LO_vertices)
    else:
        LO1_mask = np.isin(flat_idx, LO1_vertices)
        LO2_mask = np.isin(flat_idx, LO2_vertices)

    roi_colors_array = np.full(n_vertices, np.nan)
    roi_colors_array[V2_mask] = 0
    roi_colors_array[V3_mask] = 1
    roi_colors_array[V4_mask] = 2
    if atlas == "manual":
        roi_colors_array[LO_mask] = 3
    else:
        roi_colors_array[LO1_mask] = 3
        roi_colors_array[LO2_mask] = 4

    colmap = plt.cm.Set3

    # Bayesian CF modeling via MCMC
    distance_matrix = roi_submesh.dijkstra(0, 1)[0]

    import numpy as np
    import math as m
    from scipy.special import erf
    from tqdm.auto import tqdm
    import pickle
    import gc

    def normcdf(x, mu=0.0, sigma=1.0):
        return (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0)))) / 2.0

    def normpdf(x, mu=0.0, sigma=1.0):
        return (np.exp(-((x - mu) / sigma) ** 2 / 2)) / (sigma * np.sqrt(2 * np.pi))

    def sigma_from_latent(lSigma, radius=10.5, rMin=2):
        """Log transform; rMin kept for API compatibility only."""
        return np.exp(lSigma)

    def gaussian_weight(distances, lSigma, radius=10.5, rMin=0.05):
        sigma = sigma_from_latent(lSigma, radius, rMin)
        w = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        w = w / np.sum(w)
        return w.astype(np.float32)

    def beta_from_latent(lBeta):
        return np.exp(lBeta)

    def compute_cf_prediction(source_time_series, distances, lSigma, radius=10.5, rMin=0.05, betaBool=False, lBeta=None):
        w = gaussian_weight(distances, lSigma, radius, rMin)
        prediction = np.dot(source_time_series, w)
        sigma = sigma_from_latent(lSigma, radius, rMin)
        if betaBool and lBeta is not None:
            beta = beta_from_latent(lBeta)
            prediction = beta * prediction
        else:
            beta = 1.0
        return {'prediction': prediction, 'sigma': sigma, 'beta': beta, 'weights': w}

    def compute_log_likelihood(target_signal, source_time_series, distances, lSigma, radius=10.5, rMin=0.05, betaBool=False, lBeta=-5.0):
        pred = compute_cf_prediction(source_time_series, distances, lSigma, radius, rMin, betaBool, lBeta)
        if not betaBool:
            X = np.column_stack([pred['prediction'], np.ones(len(target_signal))])
            bHat = np.linalg.lstsq(X, target_signal, rcond=None)[0]
            beta = bHat[0]
            fitted_prediction = X @ bHat
        else:
            beta = pred['beta']
            fitted_prediction = pred['prediction']
        residuals = target_signal - fitted_prediction
        var_residuals = np.var(residuals)
        var_target = np.var(target_signal)
        variance_explained = 1 - (var_residuals / var_target) if var_target > 0 else 0
        mu_resid = np.mean(residuals)
        sigma_resid = np.std(residuals)
        if sigma_resid > 0:
            log_like = np.sum(np.log(normpdf(residuals, mu_resid, sigma_resid)))
        else:
            log_like = -np.inf
        # log-normal prior on sigma with soft penalty above radius
        sigma_val = pred['sigma']
        if sigma_val > radius:
            with np.errstate(divide='ignore'):
                prior_sigma = np.log(normpdf(lSigma, np.log(radius/2), 1.0)) - ((sigma_val - radius) ** 2) / (2 * radius ** 2)
        else:
            with np.errstate(divide='ignore'):
                prior_sigma = np.log(normpdf(lSigma, np.log(radius/2), 1.0))
        with np.errstate(divide='ignore'):
            prior_beta = np.log(normpdf(lBeta, -2, 5)) if betaBool else 0.0
        posterior = log_like + prior_sigma + prior_beta
        return {'log_likelihood': log_like, 'posterior': posterior, 'prior_sigma': prior_sigma,
                'prior_beta': prior_beta, 'sigma': pred['sigma'], 'beta': beta,
                'variance_explained': variance_explained, 'residuals': residuals, 'var_residuals': var_residuals}

    def propose_distance(lStepSize, maxStep):
        return np.abs(maxStep * normcdf(lStepSize) - maxStep / 2)

    def propose_center_vertex(distance_proposal, distance_matrix, current_center_idx):
        distances_from_current = distance_matrix[:, current_center_idx]
        abs_diff = np.abs(distances_from_current - distance_proposal)
        candidates = np.where(abs_diff == np.min(abs_diff))[0]
        return np.random.choice(candidates)

    def run_bayesian_cf_mcmc(target_signal, source_time_series, distance_matrix,
                              n_iter=17500, radius=10.5, rMin=0.05, betaBool=False,
                              burnIn=True, percBurnIn=10, verbose=False,
                              sigma_proposal_width=0.1, beta_proposal_width=0.1, max_step_divisor=6):
        """Metropolis-Hastings MCMC for a single target vertex."""
        n_source = source_time_series.shape[1]
        lSigma = np.log(radius / 2)
        lBeta = -5.0
        center_idx = np.random.randint(0, n_source)

        sigma_chain = np.zeros(n_iter)
        beta_chain = np.zeros(n_iter)
        center_chain = np.zeros(n_iter, dtype=int)
        variance_explained_chain = np.zeros(n_iter)
        log_likelihood_chain = np.zeros(n_iter)
        posterior_chain = np.zeros(n_iter)
        accepted = np.zeros(n_iter, dtype=bool)

        max_step = np.max(distance_matrix) / max_step_divisor
        distances_from_center = distance_matrix[:, center_idx]
        current_state = compute_log_likelihood(target_signal, source_time_series, distances_from_center,
                                               lSigma, radius, rMin, betaBool, lBeta)

        pbar = tqdm(range(n_iter), desc='MCMC', disable=not verbose, leave=False)
        for i in pbar:
            lSigma_prop = lSigma + np.random.normal(0, sigma_proposal_width)
            lBeta_prop  = lBeta  + np.random.normal(0, beta_proposal_width)
            step_proposal  = np.random.normal(0, 1)
            dist_proposal  = propose_distance(step_proposal, max_step)
            center_idx_prop = propose_center_vertex(dist_proposal, distance_matrix, center_idx)
            distances_from_prop_center = distance_matrix[:, center_idx_prop]
            proposed_state = compute_log_likelihood(target_signal, source_time_series, distances_from_prop_center,
                                                    lSigma_prop, radius, rMin, betaBool, lBeta_prop)
            log_acceptance_ratio = proposed_state['posterior'] - current_state['posterior']
            if np.random.rand() < np.exp(min(0, log_acceptance_ratio)):
                lSigma = lSigma_prop; lBeta = lBeta_prop
                center_idx = center_idx_prop
                current_state = proposed_state
                accepted[i] = True
            sigma_chain[i] = current_state['sigma']
            beta_chain[i]  = current_state['beta']
            center_chain[i] = center_idx
            variance_explained_chain[i] = current_state['variance_explained']
            log_likelihood_chain[i]  = current_state['log_likelihood']
            posterior_chain[i] = current_state['posterior']
            if (i + 1) % 100 == 0:
                pbar.set_postfix({'acc_rate': f'{np.mean(accepted[:i+1]):.3f}'})
        pbar.close()

        if burnIn and n_iter > percBurnIn:
            burn_idx = int(np.ceil(n_iter * percBurnIn / 100))
            sigma_chain = sigma_chain[burn_idx:]; beta_chain = beta_chain[burn_idx:]
            center_chain = center_chain[burn_idx:]
            variance_explained_chain = variance_explained_chain[burn_idx:]
            log_likelihood_chain = log_likelihood_chain[burn_idx:]
            posterior_chain = posterior_chain[burn_idx:]; accepted = accepted[burn_idx:]

        best_idx = np.argmax(log_likelihood_chain)
        return {'best_sigma': sigma_chain[best_idx], 'best_beta': beta_chain[best_idx],
                'best_center_idx': center_chain[best_idx],
                'best_variance_explained': variance_explained_chain[best_idx],
                'best_log_likelihood': log_likelihood_chain[best_idx],
                'best_posterior': posterior_chain[best_idx],
                'sigma_chain': sigma_chain, 'beta_chain': beta_chain,
                'center_idx_chain': center_chain,
                'variance_explained_chain': variance_explained_chain,
                'log_likelihood_chain': log_likelihood_chain,
                'posterior_chain': posterior_chain,
                'acceptance_rate': np.mean(accepted), 'n_iterations': len(sigma_chain)}

    def run_bayesian_cf_numpy(source_data, target_data, distance_matrix, target_indices, target_roi_labels,
                               n_iter=17500, radius=10.5, rMin=0.05, betaBool=False,
                               burnIn=True, percBurnIn=10, n_jobs=8, verbose=True,
                               sigma_proposal_width=0.1, beta_proposal_width=0.1, max_step_divisor=6):
        """Run MCMC for all target vertices. Returns a DataFrame with one row per vertex."""
        import time
        from joblib import Parallel, delayed
        from tqdm.auto import tqdm
        start_time = time.time()
        jobs = (
            delayed(process_single_target_mcmc_numpy)(
                target_signal=target_data[i], source_time_series=source_data,
                distance_matrix=distance_matrix, target_idx=target_indices[i],
                roi_label=target_roi_labels[i], n_iter=n_iter, radius=radius,
                rMin=rMin, betaBool=betaBool, burnIn=burnIn, percBurnIn=percBurnIn,
                sigma_proposal_width=sigma_proposal_width, beta_proposal_width=beta_proposal_width,
                max_step_divisor=max_step_divisor)
            for i in range(len(target_indices))
        )
        results_list = []
        with tqdm(total=len(target_indices), desc='Processing vertices', disable=not verbose) as pbar:
            for result in Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(jobs):
                results_list.append(result)
                pbar.update(1)
        results = [r for r in results_list if r is not None]
        results_df = pd.DataFrame(results)
        total_time = time.time() - start_time
        if verbose:
            n_failed = len(target_indices) - len(results_df)
            print(f"  done in {total_time:.1f}s ({total_time/60:.1f} min), {n_failed} failed")
            if n_jobs > 1:
                estimated_sequential_time = total_time * n_jobs
        return results_df

    def process_single_target_mcmc_numpy(target_signal, source_time_series, distance_matrix,
                                          target_idx, roi_label, n_iter, radius, rMin,
                                          betaBool, burnIn, percBurnIn,
                                          sigma_proposal_width=0.1, beta_proposal_width=0.1, max_step_divisor=6):
        try:
            mcmc_result = run_bayesian_cf_mcmc(
                target_signal=target_signal, source_time_series=source_time_series,
                distance_matrix=distance_matrix, n_iter=n_iter, radius=radius, rMin=rMin,
                betaBool=betaBool, burnIn=burnIn, percBurnIn=percBurnIn, verbose=False,
                sigma_proposal_width=sigma_proposal_width, beta_proposal_width=beta_proposal_width,
                max_step_divisor=max_step_divisor)
            return {'target_vertex_idx': target_idx, 'target_roi_label': roi_label,
                    'source_vertex_idx': int(mcmc_result['best_center_idx']),
                    'sigma_mm': float(mcmc_result['best_sigma']),
                    'beta': float(mcmc_result['best_beta']),
                    'variance_explained': float(mcmc_result['best_variance_explained']),
                    'log_likelihood': float(mcmc_result['best_log_likelihood']),
                    'posterior': float(mcmc_result['best_posterior']),
                    'acceptance_rate': float(mcmc_result['acceptance_rate']),
                    'n_iterations': int(mcmc_result['n_iterations']),
                    'center_idx_chain': mcmc_result['center_idx_chain'],
                    'sigma_chain': mcmc_result['sigma_chain'],
                    'beta_chain': mcmc_result['beta_chain'],
                    'sigma_mean': float(np.mean(mcmc_result['sigma_chain'])),
                    'sigma_std': float(np.std(mcmc_result['sigma_chain'])),
                    'sigma_ci_lower': float(np.percentile(mcmc_result['sigma_chain'], 2.5)),
                    'sigma_ci_upper': float(np.percentile(mcmc_result['sigma_chain'], 97.5)),
                    'beta_mean': float(np.mean(mcmc_result['beta_chain'])),
                    'beta_std': float(np.std(mcmc_result['beta_chain'])),
                    'beta_ci_lower': float(np.percentile(mcmc_result['beta_chain'], 2.5)),
                    'beta_ci_upper': float(np.percentile(mcmc_result['beta_chain'], 97.5))}
        except Exception as e:
            print(f"  failed for target vertex {target_idx}: {e}")
            return None

    # run Bayesian MCMC
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    pkl_file = os.path.join(results_dir, f"{h}_MCMC_CF_{task}_{atlas}_ses-{session_id}.pkl")

    # sigma_pw=0.2 confirmed best in sweep, using 0.1 per user preference; ms_div=4
    sigma_pw, beta_pw, ms_div = 0.1, 0.1, 4

    bayesian_results = run_bayesian_cf_numpy(
        source_data=source_data, target_data=target_data.T,
        distance_matrix=distance_matrix,
        target_indices=target_indices, target_roi_labels=target_roi_labels,
        n_iter=17500, radius=10.5, rMin=0.05, betaBool=True,
        burnIn=True, percBurnIn=10, n_jobs=10, verbose=True,
        sigma_proposal_width=sigma_pw, beta_proposal_width=beta_pw, max_step_divisor=ms_div
    )

    # convert chain center indices (submesh space) to flatmap and FreeSurfer native space
    _sub_to_flat = np.where(V1_mask_submesh)[0]
    _sub_to_fs   = flat_idx[_sub_to_flat]

    def _chain_ecc_polar(center_chain):
        flat_centers = _sub_to_flat[center_chain.astype(int)]
        return ecc_atlas[flat_centers], polar_atlas[flat_centers]

    source_submesh_idx = bayesian_results["source_vertex_idx"].values.astype(int)
    source_fs_idx      = _sub_to_fs[source_submesh_idx]

    best_flat_centers = _sub_to_flat[source_submesh_idx]
    ecc_mean_arr   = ecc_atlas[best_flat_centers]
    polar_mean_arr = polar_atlas[best_flat_centers]
    ecc_std_arr    = np.zeros(len(bayesian_results), dtype=float)
    polar_std_arr  = np.zeros(len(bayesian_results), dtype=float)

    _n_hemi = sub.hemis[h].vertex_count
    assert _sub_to_fs.max() > 10000, f"source_fs_idx max={_sub_to_fs.max()} -- still looks like submesh, not native FS indices"
    assert flat_idx[target_indices_full].max() > 10000, f"target_vertex_idx max={flat_idx[target_indices_full].max()} -- not native FS"

    bayesian_cf_results = {
        "source_vertex_idx":  source_fs_idx,
        "target_vertex_idx":  flat_idx[target_indices_full],
        "target_roi_label":   bayesian_results["target_roi_label"].values,
        "sigma_mean":         bayesian_results["sigma_mean"].values,
        "sigma_std":          bayesian_results["sigma_std"].values,
        "beta_mean":          bayesian_results["beta_mean"].values,
        "beta_std":           bayesian_results["beta_std"].values,
        "ecc_mean":           ecc_mean_arr,
        "ecc_std":            ecc_std_arr,
        "polar_mean":         polar_mean_arr,
        "polar_std":          polar_std_arr,
        "r2":                 bayesian_results["variance_explained"].values,
        "center_idx_chain":   bayesian_results["center_idx_chain"].values,
    }
    with open(pkl_file, "wb") as f:
        pickle.dump(bayesian_cf_results, f)
    print(f"Saved: {pkl_file}  (sub-{subject_id}, {h})")

    acceptance_rates = bayesian_results["acceptance_rate"].values
    n_low  = np.sum(acceptance_rates < 0.10)
    n_high = np.sum(acceptance_rates > 0.50)

    v1_flat_pos = np.where(V1_mask_submesh)[0]
    v1_ecc_stored = ecc_atlas[v1_flat_pos] if atlas == "benson" else ecc[v1_flat_pos]
    all_results[atlas] = {
        'bayesian_cf_results': bayesian_cf_results,
        'acceptance_rates':    acceptance_rates,
        'ecc_range':           ecc_range,
        'v1_ecc':              v1_ecc_stored,
    }

    r2_threshold = 0.1
    if atlas == "benson":
        ecc_source   = ecc_atlas
        polar_source = polar_atlas
    else:
        ecc_source   = ecc
        polar_source = polar

    Grid_CF = {'centers': gf.gridsearch_params[:, 0].astype(int),
               'sigma': gf.gridsearch_params[:, 1],
               'r2': gf.gridsearch_r2, 'opType': 'Grid search'}
    if gradescent_results is not None:
        GradientDescent_CF = {'centers': gradescent_results['vertex_indices'].astype(int),
                              'sigma': gradescent_results['sigma'],
                              'r2': gradescent_results['r2'], 'opType': 'Gradient descent'}
    else:
        GradientDescent_CF = Grid_CF

    Bayesian_CF = {'centers': source_submesh_idx,
                   'sigma': bayesian_cf_results['sigma_mean'],
                   'r2': bayesian_cf_results['r2'], 'opType': 'Bayesian optimization'}

    n_vertices_fm = len(flat_idx)
    cf_sigma_fm     = np.full(n_vertices_fm, np.nan)
    cf_sigma_unc_fm = np.full(n_vertices_fm, np.nan)
    cf_beta_fm      = np.full(n_vertices_fm, np.nan)
    cf_beta_unc_fm  = np.full(n_vertices_fm, np.nan)
    cf_ecc_fm       = np.full(n_vertices_fm, np.nan)
    cf_polar_fm     = np.full(n_vertices_fm, np.nan)
    cf_sigma_fm[target_indices_full]     = bayesian_cf_results["sigma_mean"]
    cf_sigma_unc_fm[target_indices_full] = bayesian_cf_results["sigma_std"]
    cf_beta_fm[target_indices_full]      = bayesian_cf_results["beta_mean"]
    cf_beta_unc_fm[target_indices_full]  = bayesian_cf_results["beta_std"]
    cf_ecc_fm[target_indices_full]       = bayesian_cf_results["ecc_mean"]
    cf_polar_fm[target_indices_full]     = bayesian_cf_results["polar_mean"]

    v1_submesh_flat_fm = np.where(V1_mask_submesh)[0]
    v1_sel_counts = np.zeros(len(v1_submesh_flat_fm), dtype=np.float32)
    for _idx in source_submesh_idx:
        v1_sel_counts[_idx] += 1
    v1_counts_flat_fm = np.full(n_vertices_fm, np.nan)
    v1_counts_flat_fm[v1_submesh_flat_fm] = v1_sel_counts

    all_results[atlas] = {'bayesian_cf_results': bayesian_cf_results, 'acceptance_rates': bayesian_results["acceptance_rate"].values, 'ecc_range': ecc_range, 'v1_ecc': v1_ecc_stored, 'ecc_source': ecc_source, 'polar_source': polar_source, 'submesh_indices': submesh_indices, 'target_roi_mask': target_roi_mask, 'target_indices_full': target_indices_full, 'flatmaps': flatmaps, 'flat_idx': flat_idx, 'GradientDescent_CF': GradientDescent_CF, 'Bayesian_CF': Bayesian_CF, 'ret_ecc_compare': ecc_source[submesh_indices], 'ret_pol_compare': polar_source[submesh_indices], 'colors_ecc': colors_ecc, 'colors_polar': colors_polar, 'r2_threshold': r2_threshold, 'cf_sigma_fm': cf_sigma_fm, 'cf_sigma_unc_fm': cf_sigma_unc_fm, 'cf_beta_fm': cf_beta_fm, 'cf_beta_unc_fm': cf_beta_unc_fm, 'cf_ecc_fm': cf_ecc_fm, 'cf_polar_fm': cf_polar_fm, 'v1_counts_flat_fm': v1_counts_flat_fm, 'v1_submesh_flat_fm': v1_submesh_flat_fm, 'v1_sel_counts': v1_sel_counts, 'Grid_CF': Grid_CF}

    # save CF maps back to FreeSurfer native space as MGZ
    import nibabel as nib
    cortex_idx = flatmaps[h].prop('index')
    n_hemi_vertices = sub.hemis[h].vertex_count

    cf_ecc_native   = np.full(n_hemi_vertices, np.nan)
    cf_sigma_native = np.full(n_hemi_vertices, np.nan)
    cf_ecc_native[cortex_idx]   = cf_ecc_fm
    cf_sigma_native[cortex_idx] = cf_sigma_fm

    ref_mgz = nib.load(os.path.join(fs_subject_path, "mri", "orig.mgz"))
    mgh_hdr = nib.freesurfer.mghformat.MGHHeader()
    mgh_hdr.set_data_shape(cf_ecc_native.shape)
    mgh_hdr.set_data_dtype(np.float32)
    cf_ecc_mgz   = nib.MGHImage(cf_ecc_native.astype(np.float32),   affine=ref_mgz.affine, header=mgh_hdr)
    cf_sigma_mgz = nib.MGHImage(cf_sigma_native.astype(np.float32), affine=ref_mgz.affine, header=mgh_hdr)

    nib.save(cf_ecc_mgz,   os.path.join(surf_dir, f"{h}_CF_eccentricity_{task}_{atlas}_ses-{session_id}.mgz"))
    nib.save(cf_sigma_mgz, os.path.join(surf_dir, f"{h}_CF_sigma_{task}_{atlas}_ses-{session_id}.mgz"))

    # V1 selection count map: how many target vertices chose each V1 vertex as best CF center
    v1_submesh_flat = np.where(V1_mask_submesh)[0]
    n_v1 = len(v1_submesh_flat)
    v1_selection_counts = np.zeros(n_v1, dtype=np.float32)
    for idx in source_submesh_idx:
        v1_selection_counts[idx] += 1

    v1_counts_flat = np.full(len(flat_idx), np.nan)
    v1_counts_flat[v1_submesh_flat] = v1_selection_counts
    v1_counts_native = np.full(n_hemi_vertices, np.nan)
    v1_counts_native[cortex_idx] = v1_counts_flat

    mgh_hdr_counts = nib.freesurfer.mghformat.MGHHeader()
    mgh_hdr_counts.set_data_shape(v1_counts_native.shape)
    mgh_hdr_counts.set_data_dtype(np.float32)
    v1_counts_mgz = nib.MGHImage(v1_counts_native.astype(np.float32), affine=ref_mgz.affine, header=mgh_hdr_counts)
    nib.save(v1_counts_mgz, os.path.join(surf_dir, f"{h}_V1_selection_count_{task}_{atlas}_ses-{session_id}.mgz"))

    all_results[atlas]['v1_ecc_vals_for_hist']   = ecc_atlas[v1_submesh_flat]
    all_results[atlas]['v1_sel_counts_for_hist']  = v1_selection_counts.copy()
