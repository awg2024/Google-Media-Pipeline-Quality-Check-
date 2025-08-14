import os
import numpy as np
import pandas as pd
import yaml
from itertools import combinations
from pathlib import Path
from scipy.signal import savgol_filter
from automatic_visualisation import automatic_visualisation
import math
import random
from collections import defaultdict
from accuracy_matrix import compute_accuracy_matrix 
from tabulate import tabulate
import numpy as np
from itertools import combinations



### ========== User Input ==========

# def ask_user():

#     fingertip_scheme = [
#     "RIGHT_THUMB", "RIGHT_INDEX", "RIGHT_MIDDLE", "RIGHT_RING", "RIGHT_PINKY"
#     ]

#     hand_scheme = [
#         "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
#         "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
#         "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
#         "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
#         "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
#         "THIRD_THUMB_MCP", "THIRD_THUMB_PIP", "THIRD_THUMB_DIP", "THIRD_THUMB_TIP"
#     ]

#     print("Welcome to the Quality Check for the Third Thumb Nullspace Experiment. Please ensure your markerlessTrackingEnv conda environment is up to date.")
#     participant = input("1) Participant number (e.g., 01): ").strip()
#     day = input("4) Day to analyze?: ").strip()
#     overall_session = input("2) Run QC on the entire session? (y/n): ").strip().lower()
    
#     if overall_session == 'y':
#         return participant, overall_session, None, day, None, None

#     task = input("3) Task name: ").strip()
#     trial = input("5) Trial to analyze?: ").strip()
#     advanced = input("6) Advanced mode – select joints or schemes? (y/n): ").strip().lower()

#     if advanced == 'y':
#         choice = input("Enter 'HAND', 'FINGERTIPS', or comma‑separated joints: ").strip().upper()
#         if choice == "HAND":
#             joints = hand_scheme
#         elif choice == "FINGERTIPS":
#             joints = fingertip_scheme
#         else:
#             joints = [j.strip() for j in choice.split(',') if j.strip()]
#     else:
#         joints = None

#     return participant, overall_session, task, day, trial, joints

### ========== Utility Functions ==================

def extract_joint_names(df):
    return sorted({col.rsplit('_', 1)[0] for col in df.columns if col.endswith('_x')})


# def random_selection(trials, n_per_task, seed=42):
#     """
#     Selects up to `n_per_task` trials per task from the provided list of (day, task, trial) tuples.
#     Returns a shuffled list of selected trials.

#     Parameters:
#         trials (List[Tuple[str, str, str]]): All trials (day, task, trial)
#         n_per_task (int): Number of trials to select per task
#         seed (int): Random seed for reproducibility

#     Returns:
#         List[Tuple[str, str, str]]: Evenly distributed selected trials
#     """
#     random.seed(seed)
#     task_groups = defaultdict(list)
    
#     for day, task, trial in trials:
#         task_groups[task].append((day, task, trial))

#     selected = []
#     for task, t_trials in task_groups.items():
#         if len(t_trials) <= n_per_task:
#             selected.extend(t_trials)
#         else:
#             selected.extend(random.sample(t_trials, n_per_task))

#     random.shuffle(selected)
#     return selected


### ========= FIRST ROUND QC Calculation Functions ===========

def reprojection_calculations(df, threshold):
    """
    Calculates the maximum reprojection error per row and flags any rows exceeding the threshold.
    
    Parameters:
        df : pandas.DataFrame
            DataFrame containing reprojection error columns for joints (ending with '_error').
        threshold : float
            Maximum allowed reprojection error. Rows exceeding this will be flagged.
    
    Returns:
        max_error : np.ndarray
            Maximum reprojection error per row.
        flagged_idxs : np.ndarray or None
            Indices of rows where the error exceeds the threshold. None if no rows flagged.
    """
    # Find all error columns
    error_cols = [col for col in df.columns if col.endswith("_error")]
    if not error_cols:
        return np.zeros(len(df)), None
    
    # Compute per-row maximum across error columns
    max_error = df[error_cols].fillna(0).to_numpy().max(axis=1)
    
    # Flag rows exceeding threshold
    flagged_idxs = np.where(max_error > threshold)[0]
    
    # Return None if nothing is flagged
    if flagged_idxs.size == 0:
        flagged_idxs = None
    
    return max_error, flagged_idxs



### ========== SECOND ROUND QC Calculation Functions ==========
# Here we are not utilising the coordinates in a real-world format since these coordinates are meaningless. 
# 1. Jitter
# 2. Aperature - needs to absolulte vector distance calculated. 
# 3. Cross corr vel and acc

def compute_jitter(df, joints, jit_conf, window=11, poly=3):
    flagged = []
    for j in joints:
        entry = jit_conf.get(j)
        if not entry or 'max' not in entry:
            continue
        thresh = entry['max']
        try:
            x = df[f"{j}_x"].interpolate().ffill().bfill().to_numpy()
            y = df[f"{j}_y"].interpolate().ffill().bfill().to_numpy()
            z = df[f"{j}_z"].interpolate().ffill().bfill().to_numpy()
            if len(x) < window:
                continue
            xs = savgol_filter(x, window, poly)
            ys = savgol_filter(y, window, poly)
            zs = savgol_filter(z, window, poly)
            jitter = np.abs(x-xs) + np.abs(y-ys) + np.abs(z-zs)
            idxs = np.where(jitter > thresh)[0]
            flagged += [(j, i, "High Jitter", jitter[i], thresh) for i in idxs]
        except KeyError:
            pass
    return flagged

def compute_aperature(df, scheme, ape_conf):
    flagged = []

    for chain in scheme:
        for j1, j2 in combinations(chain, 2):
            key = f"{j1}_to_{j2}"
            entry = ape_conf.get(key)
            if not entry or 'max' not in entry:
                continue
            thresh = entry['max']
            
            try:
                # Stack positions into arrays of shape (n_samples, 3)
                p1 = df[[f"{j1}_x", f"{j1}_y", f"{j1}_z"]].to_numpy()
                p2 = df[[f"{j2}_x", f"{j2}_y", f"{j2}_z"]].to_numpy()

                # Vectorized distance calculation
                d = np.linalg.norm(p1 - p2, axis=1)

                # Get all indices exceeding threshold
                idxs = np.flatnonzero(d > thresh)

                # Use NumPy to create the flagged entries efficiently
                if idxs.size > 0:
                    flagged_entries = np.column_stack((
                        np.full_like(idxs, key, dtype=object),
                        idxs,
                        np.full_like(idxs, "Aperture Exceeded", dtype=object),
                        d[idxs],
                        np.full_like(idxs, thresh, dtype=float)
                    ))
                    flagged.extend(map(tuple, flagged_entries))
            except KeyError:
                continue

    return flagged

def compute_crosscorr(df, joints, window=30, corr_thresh=0.8):
    """
    Flags joints where velocity and acceleration are highly correlated over a sliding window,
    which can indicate systematic tracking issues or jitter patterns.
    
    Args:
        df: DataFrame with joint positions.
        joints: List of joint names.
        window: Number of frames to compute cross-correlation over (sliding window).
        corr_thresh: Threshold for absolute correlation to flag issues.
    
    Returns:
        flagged: List of tuples (joint, frame_idx, issue, corr_value, threshold)
    """
    
    # calculations
    # 60 fps 
    # cross corr every 50 ms / 0.5 seconds 
    
    flagged = []

    for joint in joints:
        try:
            pos = df[[f"{joint}_x", f"{joint}_y", f"{joint}_z"]].interpolate().ffill().bfill().to_numpy()
            if len(pos) < window + 2:
                continue

            # Compute velocity and acceleration magnitudes
            # Computing absolute vectorised values for both. 
            velocity = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            acceleration = np.linalg.norm(np.diff(np.diff(pos, axis=0), axis=0))

            # Pad to match lengths
            velocity = np.pad(velocity, (1,0), mode='constant')
            acceleration = np.pad(acceleration, (2,0), mode='constant')

            # Sliding window cross-correlation
            for i in range(len(velocity) - window + 1):
                v_win = velocity[i:i+window]
                a_win = acceleration[i:i+window]
                if np.std(v_win) == 0 or np.std(a_win) == 0:
                    continue  # skip if flat
                corr = np.corrcoef(v_win, a_win)[0,1]
                if abs(corr) > corr_thresh:
                    flagged.append((joint, i, "High Vel-Acc Correlation", round(corr,3), corr_thresh))

        except KeyError:
            continue

    return flagged


### ========== QC Functions to Run over Multiple Folders - Wide-Spread selection ==========

def process_folder(folder, participant, task, day, trial, joints):
    
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
    
    scheme = [
        ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_PINKY"],
        ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"],
        ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_THUMB"],
        ["WRIST", "THUMB_TIP"],
        ["WRIST", "INDEX_FINGER_TIP"],
        ["WRIST", "THIRD_THUMB_TIP"]
    ]

    config_file_path = r"\\cbsu\data\Group\Plasticity\Projects\Null_space_markerless_tracking\Quality_Check_Pose_Estimation\config_thresholds.yml"
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if not folder.exists():
        print(f"Folder missing: {folder}")
        return
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        print(f"No CSVs in: {folder}")
        return

    rows = []
    for f in csvs:
        df = pd.read_csv(f)
        js = joints if joints else extract_joint_names(df)

        save_kinematic_metrics(df, folder, js, participant, task, day)

        # ---- First step QC: reprojection ----
        reprojection_results = reprojection_calculations(df, js, config['REPROJECTION'])

        rows += [(f.name, *r) for r in reprojection_results]

        # ---- Only run other QC metrics if reprojection passes ----

        if not reprojection_results:  # none means no reprojection issues
            rows += [(f.name, *r) for r in compute_jitter(df, js, config['JITTER'])]
            rows += [(f.name, *r) for r in compute_aperature(df, scheme, config['APERATURE'])]
            rows += [(f.name, *r) for r in compute_crosscorr(df, js, config['CROSSCORR'])]
        else:
            print(f"Skipping jitter/aperature/crosscorr for {f.name} due to reprojection check has failed.")

        
    if rows:
        # Create QC output folder
        qc_folder = folder / "Quality_check_output"
        qc_folder.mkdir(exist_ok=True)

        # Save full CSV report anyway (optional)
        df_out = pd.DataFrame(rows, columns=['File','Joint','Frame','Issue','Value','Threshold'])
        out = qc_folder / f"QC_Report_{participant}_{task}_{day}.csv"
        df_out.to_csv(out, index=False)

        # Stepwise: only report the first type of error detected
        first_issue = df_out['Issue'].iloc[0]
        session_label = f"{participant}_{task}_{day}"
        print(f"QC failed for session: {session_label} -> {first_issue}")

        print(f"Detailed QC report saved to: {out}")

    else:
        print(f"No issues in folder: {folder}")



# saving kinematic results to pose_3d folder. 

def save_kinematic_metrics(df, output_folder, joints, participant, task, day):
    metrics = []

    for joint in joints:
        try:
            pos = df[[f"{joint}_x", f"{joint}_y", f"{joint}_z"]].interpolate().ffill().bfill().to_numpy()
            if len(pos) < 3:
                continue
            velocity = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            acceleration = np.linalg.norm(np.diff(np.diff(pos, axis=0), axis=0), axis=1)

            velocity = np.pad(velocity, (1, 0), mode='constant')
            acceleration = np.pad(acceleration, (2, 0), mode='constant')

            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]

            xs = savgol_filter(x, 11, 3)
            ys = savgol_filter(y, 11, 3)
            zs = savgol_filter(z, 11, 3)

            jitter = np.abs(x - xs) + np.abs(y - ys) + np.abs(z - zs)

            error_col = f"{joint}_error"
            reproj = df[error_col].fillna(0).to_numpy() if error_col in df.columns else np.zeros(len(df))

            for i in range(len(df)):
                metrics.append({
                    'Frame': i,
                    'Joint': joint,
                    'Velocity': velocity[i],
                    'Acceleration': acceleration[i],
                    'Jitter': jitter[i],
                    'ReprojectionError': reproj[i]
                })

        except KeyError:
            continue

    df_out = pd.DataFrame(metrics)
    kin_folder = output_folder / "quality_check_output"
    kin_folder.mkdir(exist_ok=True)
    outfile = kin_folder / f"Kinematics_{participant}_{task}_{day}.csv"
    df_out.to_csv(outfile, index=False)
    print(f"Kinematic data saved to: {outfile}")
