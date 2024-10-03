#imports
import numpy as np
import pandas as pd

#Read the trajectory file and ensure timestamps are read as floats - will be using from utils.py
def read_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype={0: float})
    data.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return data

#Get the nearest unique timestamps between two timestamp arrays
def find_nearest_unique_timestamps(reference_timestamps, target_timestamps):
    # Array to store the indices of the nearest unique timestamps
    nearest_indices = []
    used_indices = set()
    
    for ref_time in reference_timestamps:
        # Calculate absolute differences between the reference timestamp and all target timestamps
        differences = np.abs(target_timestamps - ref_time)
        
        # Mask out the already used indices by setting their differences to a large number
        differences[list(used_indices)] = np.inf
        
        # Find the index of the minimum difference
        nearest_index = np.argmin(differences)
        nearest_indices.append(nearest_index)
        
        # Mark this index as used
        used_indices.add(nearest_index)
    
    return nearest_indices

#Synchronize the ground truth trajectory with the SLAM trajectory
def synchronize_trajectories(trajectory_gt, trajectory_openvslam):
    # Extract timestamps
    timestamps_gt = trajectory_gt['timestamp'].values
    timestamps_openvslam = trajectory_openvslam['timestamp'].values

    # Find nearest timestamps in OpenVSLAM for GT timestamps
    nearest_indices = find_nearest_unique_timestamps(timestamps_openvslam, timestamps_gt)

    # Get the synchronized GT based on the nearest timestamps
    synchronized_gt = trajectory_gt.iloc[nearest_indices].reset_index(drop=True)
    return synchronized_gt

#Save synchronized trajectories - will be using from utils.py
def save_synchronized_trajectory(trajectory, output_path):
    trajectory.to_csv(output_path, sep=' ', header=False, index=False)
    print(f"Synchronized trajectory saved to {output_path}")


def main():
    openvslam_path = '/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt'
    gt_path = '/home/sridhar03/Downloads/new_traj_sync/mh05/data.tum'
    output_path = '/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_gt.txt'

    trajectory_openvslam = read_trajectory(openvslam_path)
    trajectory_gt = read_trajectory(gt_path)

    #Call to algo - Synchronize trajectories
    synchronized_gt = synchronize_trajectories(trajectory_gt, trajectory_openvslam)

    #Save the synchronized trajectory - will be using from utils.py
    save_synchronized_trajectory(synchronized_gt, output_path)

if __name__ == "__main__":
    main()
