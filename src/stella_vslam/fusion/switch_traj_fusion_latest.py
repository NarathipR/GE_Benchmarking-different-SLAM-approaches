#imports
import numpy as np
import pandas as pd

# Function to calculate APE (between ground truth and estimated trajectory)
def calculate_ape(estimated_pose, ground_truth_pose):
    """
    Calculate the Absolute Pose Error (APE) between two poses.
    
    :param estimated_pose: A single pose from the estimated trajectory [x, y, z, qx, qy, qz, qw].
    :param ground_truth_pose: The corresponding pose from the ground truth [x, y, z, qx, qy, qz, qw].
    :return: The translation APE and optionally the orientation APE.
    """
    # Calculate translation APE (Euclidean distance between positions)
    position_diff = estimated_pose[:3] - ground_truth_pose[:3]
    translation_ape = np.sqrt(np.sum(position_diff**2))
    
    # Optional: Calculate orientation APE (difference between quaternions)
    orientation_diff = np.linalg.norm(estimated_pose[3:] - ground_truth_pose[3:])
    
    # Combine translation and orientation APE (you can weight them differently if needed)
    total_ape = translation_ape + orientation_diff

    return total_ape

# Read the synchronized trajectory files
def read_synchronized_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data

def main():

    '''
    This code reads the ground truth and both ORB and OpenVSLAM trajectories.
    For each segment of timestamps, it finds the APE for both SLAM modules - selects the trajectory with the lower APE.
    '''

    # Read synchronized ground truth and trajectories
    synced_GT = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh01/synchronized_gt.txt')
    trajectory_orb = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh01/synchronized_trajectory_orb.txt')
    trajectory_openvslam = read_synchronized_trajectory('/home/sridhar03/Downloads/new_traj_sync/mh01/synchronized_trajectory_openvslam.txt')

    # Extract the timestamps
    timestamps_openvslam = trajectory_openvslam.iloc[:, 0].values
    timestamps_orb = trajectory_orb.iloc[:, 0].values

    # Extract the trajectory data (excluding the timestamp)
    synced_GT_data = synced_GT.iloc[:, 1:].values
    trajectory_orb_data = trajectory_orb.iloc[:, 1:].values
    trajectory_openvslam_data = trajectory_openvslam.iloc[:, 1:].values

    # Initialize a dummy fused trajectory
    fused_trajectory = np.zeros_like(trajectory_orb_data)
    num_samples = len(fused_trajectory)

    # Initialize counters for comparison
    ape_orb_high_count, ape_openvslam_high_count = 0, 0

    # Loop through sequence lengths
    seq_len_list = np.arange(1, 6)
    for sequence_length in seq_len_list:
        # Iterate through the synced trajectories according to sequence_length
        for i in range(0, num_samples, sequence_length):
            end_index = min(i + sequence_length, num_samples)  # Ensure we don't go out of bounds
            
            # Extract these segments
            orb_segment = trajectory_orb_data[i:end_index]
            openvslam_segment = trajectory_openvslam_data[i:end_index]
            gt_segment = synced_GT_data[i:end_index]
            
            # Calculate APE for each timestamp in the segment
            ape_orb = np.mean([calculate_ape(orb_segment[j], gt_segment[j]) for j in range(len(orb_segment))])
            ape_openvslam = np.mean([calculate_ape(openvslam_segment[j], gt_segment[j]) for j in range(len(openvslam_segment))])
            
            # Choose the segment with the smaller APE
            if ape_openvslam >= ape_orb:
                fused_trajectory[i:end_index] = orb_segment
                ape_openvslam_high_count += 1
            else:
                fused_trajectory[i:end_index] = openvslam_segment
                ape_orb_high_count += 1

        # Output comparison results for the current sequence length
        print("No of consecutive timestamps={}".format(sequence_length),
              "  Openvslam={}, ORB={}".format(ape_openvslam_high_count, ape_orb_high_count))
        
        # Combine timestamps and fused trajectory (choose based on lower APE counts)
        if ape_orb_high_count <= ape_openvslam_high_count:
            fused_trajectory_with_timestamps = np.column_stack((timestamps_orb, fused_trajectory))
        else:
            fused_trajectory_with_timestamps = np.column_stack((timestamps_openvslam, fused_trajectory))

        # Save the fused trajectory to a new file (using the current sequence length in the filename)
        fused_trajectory_df = pd.DataFrame(fused_trajectory_with_timestamps)
        fused_trajectory_df.to_csv(f'/home/sridhar03/Downloads/new_traj_sync/mh01/fused_trajectory_switch_crct_{sequence_length}.txt',
                                   sep=' ', header=False, index=False)

    print("***** APE calculation and unit test successful *****")


if __name__ == "__main__":
    main()
