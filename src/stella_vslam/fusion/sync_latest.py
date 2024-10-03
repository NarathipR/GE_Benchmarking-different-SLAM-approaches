#imports
import numpy as np
import pandas as pd

#Read the trajectory file - will be using from utils.py
def read_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype={0: float})
    data.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return data


#Find the nearest unique timestamps between two arrays
def find_nearest_unique_timestamps(reference_timestamps, target_timestamps):
    nearest_indices = []
    used_indices = set()
    
    for ref_time in reference_timestamps:
        #Calculate abs difference between the reference timestamp and all target timestamps
        differences = np.abs(target_timestamps - ref_time)
        
        #this masks out the already used indices by setting their differences to larger number
        differences[list(used_indices)] = np.inf
        
        #Find the index of the minimum difference
        nearest_index = np.argmin(differences)
        nearest_indices.append(nearest_index)
        
        #this will mark this index as used
        used_indices.add(nearest_index)
    
    return nearest_indices

#Generalize ORB-SLAM3 timestamps to 10 digits - an attempt to generalize issue with different timestamp recording in diff algos
def generalize_timestamps(trajectory_orb):
    trajectory_orb['timestamp'] = trajectory_orb['timestamp'].apply(
        lambda x: x / 1e9 if len(str(int(x))) > 10 else x
    )
    return trajectory_orb

#Synchronize the ORB and OpenVSLAM trajectories
def synchronize_trajectories(trajectory_orb, trajectory_openvslam):
    #Extract timestamps - without the position/orientation data
    timestamps_orb = trajectory_orb['timestamp'].values
    timestamps_openvslam = trajectory_openvslam['timestamp'].values

    #Find nearest timestamps in OpenVSLAM for ORB-SLAM3 timestamps
    nearest_indices = find_nearest_unique_timestamps(timestamps_orb, timestamps_openvslam)

    #Get synchronized OpenVSLAM trajectory and reset indices for both
    synchronized_openvslam = trajectory_openvslam.iloc[nearest_indices].reset_index(drop=True)
    synchronized_orb = trajectory_orb.reset_index(drop=True)

    return synchronized_orb, synchronized_openvslam

#Save synchronized trajectories to files - will be using from utils.py
def save_synchronized_trajectories(synchronized_orb, synchronized_openvslam, orb_output_path, openvslam_output_path):
    synchronized_orb.to_csv(orb_output_path, sep=' ', header=False, index=False)
    synchronized_openvslam.to_csv(openvslam_output_path, sep=' ', header=False, index=False)
    print(f"Synchronized ORB saved to {orb_output_path}")
    print(f"Synchronized OpenVSLAM saved to {openvslam_output_path}")



def main():
    #File paths - can be passed as arguments for flexibility
    orb_path = '/home/sridhar03/Downloads/new_traj_sync/v101/trajectory_orb.txt'
    openvslam_path = '/home/sridhar03/Downloads/new_traj_sync/v101/frame_trajectory.txt'
    orb_output_path = '/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_orb.txt'
    openvslam_output_path = '/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt'

    #reading and preprocessing timestamps    
    trajectory_orb = read_trajectory(orb_path)
    trajectory_openvslam = read_trajectory(openvslam_path)

    trajectory_orb = generalize_timestamps(trajectory_orb)

    #Synchronize trajs
    synchronized_orb, synchronized_openvslam = synchronize_trajectories(trajectory_orb, trajectory_openvslam)

    #Save synchronized trajectories - will be using from utils.py
    save_synchronized_trajectories(synchronized_orb, synchronized_openvslam, orb_output_path, openvslam_output_path)

if __name__ == "__main__":
    main()
