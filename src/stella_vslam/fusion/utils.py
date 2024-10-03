import numpy as np 
import pandas as pd

def read_synchronized_trajectory(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    return data

def extract_timestamps(trajectory):
    
    '''
    Extract the timestamps from OpenVSLAM/ORB or any trajectory input
    Format requirement in trajectory file: 
    timestamp,x,y,z,qx,qy,qz,qw
    '''
    timestamps_trajectory = trajectory.iloc[:, 0].values
    return timestamps_trajectory

def extract_trajectory_without_timestamp(trajectory):

    '''
    Extract the trajectory data (excluding the timestamp)
    Format requirement in trajectory file: 
    timestamp,x,y,z,qx,qy,qz,qw
    '''

    trajectory_data = trajectory.iloc[:, 1:].values