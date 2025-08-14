import os
import numpy as np
import pandas as pd
import yaml
from itertools import combinations
from pathlib import Path
from scipy.signal import savgol_filter
import math
import random
from collections import defaultdict

# function script

from pose_qc_utils import( 
    ask_user,
    extract_joint_names,
    random_selection,
    extract_reprojection_error,
    joint_velocity_spikes,
    acceleration_jerk_flags,
    keypoint_jitter_flags,
    aperture_flags,
    process_folder,
    distribution_flags,
    time_constraint_flags
)
from automatic_visualisation import automatic_visualisation



### ========== Config ==========

scheme = [
    ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_PINKY"],
    ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"],
    ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_THUMB"],
    ["WRIST", "THUMB_TIP"],
    ["WRIST", "INDEX_FINGER_TIP"],
    ["WRIST", "THIRD_THUMB_TIP"]
]

third_thumb_chain = [
    "THIRD_THUMB_MCP",
    "THIRD_THUMB_PIP",
    "THIRD_THUMB_DIP",
    "THIRD_THUMB_TIP"
]

fingertip_scheme = [
    "RIGHT_THUMB", "RIGHT_INDEX", "RIGHT_MIDDLE", "RIGHT_RING", "RIGHT_PINKY"
]

hand_scheme = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    "THIRD_THUMB_MCP", "THIRD_THUMB_PIP", "THIRD_THUMB_DIP", "THIRD_THUMB_TIP"
]

TRIALS = ["Coins", "Eggs_First", "Eggs_Second", "Grasp", "GTK", "Jenga", "Jenga_Standing",
          "Pegs", "Pegs_Standing", "SFO", "Tapes", "Tapes_Standing"]

TRIAL_NOS = [1, 2, 3, 4, 5]


### ========= Load Threshold Config ========

config_file_path = r"\\cbsu\data\Group\Plasticity\Projects\Null_space_markerless_tracking\Quality_Check_Pose_Estimation\config_thresholds.yml"
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)


required_keys = ['VELOCITY', 'ACCELERATION', 'JITTER', 'APERATURE', 'REPROJECTION']

for key in required_keys:
    if key not in config:
        raise KeyError(f"Missing top-level config section: '{key}'")


### ========== Main Running Script ==============

from pathlib import Path

def run_pose_qc():

    print("Starting pose QC script.")

    # --- Hard-coded settings ---
    participant = ["PT_15"]  # e.g., PT_08, PT_10
    
    day = ["Day_01"]         # e.g., Day_02, Day_05
    
    automatic_visualisation = ["y"]   # yes or no
    
    joints = None          # specify which joints to analyse. None will analyse all of them? 

    base = Path(r"\\cbsu\data\Group\Plasticity\Projects\Null_space_markerless_tracking\Video_for_test")

    # --- Automatically find all trial folders ---
    session_folder = base / f"{participant}/{participant}_{day}"
    trial_folders = sorted([f for f in session_folder.glob("*/all_3d/2001-01-01/pose-3d") if f.is_dir()])

    print(f"Found {len(trial_folders)} trials for {participant}, {day}.")

    # --- Process each trial ---
    for fold in trial_folders:
        # Extract task and trial from folder path
        parts = fold.parts
        task_trial = parts[-5]  # adjust index based on folder depth
        
        if "_" in task_trial:
            task, trial = task_trial.rsplit("_", 1) # splitting it out (e.g., Pegs_1 = trial = 1. )
        else:
            task = task_trial
            trial = "1"  # fallback

        print(f"Processing trial: Task={task}, Trial={trial}")
        
        process_folder(fold, participant, task, day, trial, joints)

        if automatic_visualisation.lower() == "y":
            automatic_visualisation(fold, participant, day, task, trial)

    print(" Media pipeline quality check completed.")

if __name__ == "__main__":
    run_pose_qc()
