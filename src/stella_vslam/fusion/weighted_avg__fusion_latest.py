#imports
import numpy as np
import pandas as pd

#Calculate the weights based on ATE (Absolute Trajectory Error)
def calculate_weights(ate_orb, ate_openvslam):
    #Calc weights based on ATE
    weight_orb = 1 / ate_orb
    weight_openvslam = 1 / ate_openvslam

    #Normalize weights
    total_weight = weight_orb + weight_openvslam
    normalized_weight_orb = weight_orb / total_weight
    normalized_weight_openvslam = weight_openvslam / total_weight

    print("Normalized weights of ORB: {}, OpenVSLAM: {}".format(normalized_weight_orb, normalized_weight_openvslam))
    return normalized_weight_orb, normalized_weight_openvslam

#Read synchronized trajectory files - will be using from utils.py
def read_synchronized_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data

#Fuse trajectories using weighted average - based on ATE
def fuse_trajectories(trajectory_orb, trajectory_openvslam, weight_orb, weight_openvslam):
    # Compute the fused trajectory using weighted average
    fused_trajectory = (weight_orb * trajectory_orb + weight_openvslam * trajectory_openvslam)
    return fused_trajectory

#Save fused trajectory - will be using from utils.py     
def save_fused_trajectory(timestamps, fused_trajectory, output_path):
    # Append the timestamps to the fused trajectory
    fused_trajectory_with_timestamps = np.column_stack((timestamps, fused_trajectory))

    # Save the synced trajectories
    fused_trajectory_df = pd.DataFrame(fused_trajectory_with_timestamps)
    fused_trajectory_df.to_csv(output_path, sep=' ', header=False, index=False)
    print(f"Fused trajectory saved to {output_path}")


def main():
    #ATE values for ORB-SLAM3 and OpenVSLAM - got after EVO tests
    ate_orb = 0.147318
    ate_openvslam = 0.049798

    #Calling the main algo - Calculate weights based on ATE
    normalized_weight_orb, normalized_weight_openvslam = calculate_weights(ate_orb, ate_openvslam)

    # Read the synchronized trajectory files
    trajectory_orb = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_orb.txt')
    trajectory_openvslam = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh05/synchronized_trajectory_openvslam.txt')

    #Extract only the trajectory data (excluding timestamps) - will be using from utils.py
    trajectory_orb_data = trajectory_orb.iloc[:, 1:].values
    trajectory_openvslam_data = trajectory_openvslam.iloc[:, 1:].values

    #Extract only the timestamps from OpenVSLAM trajectory - will be using from utils.py
    timestamps_openvslam = trajectory_openvslam.iloc[:, 0].values

    assert trajectory_orb_data.shape == trajectory_openvslam_data.shape, "Trajectory shapes do not match"

    #this fuses the trajectories using the calculated weights
    fused_trajectory = fuse_trajectories(trajectory_orb_data, trajectory_openvslam_data, normalized_weight_orb, normalized_weight_openvslam)

    #Save the fused trajectory  - will be using from utils.py
    save_fused_trajectory(timestamps_openvslam, fused_trajectory, '/home/sridhar03/Downloads/new_traj_sync/mh05/fused_trajectory.txt')

if __name__ == "__main__":
    main()
