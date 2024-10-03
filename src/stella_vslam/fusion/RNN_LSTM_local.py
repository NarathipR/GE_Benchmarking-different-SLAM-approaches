import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split

#################
#to get GT
def read_ground_truth(file_path):
    data = np.loadtxt(file_path)
    return data

ground_truth = read_ground_truth('ground_truth.txt')
print(ground_truth[:10])
###################


#################
#to get the SLAM trajectory outputs
def read_traj_outputs(file_path):
    data = np.loadtxt(file_path)
    return data

poses_orb = read_traj_outputs('traj_ORB3.txt')
poses_openvslam = read_traj_outputs('traj_OpenVSLAM.txt')
print(poses_orb[:10])
print(poses_openvslam[:10])
################

train_input_traj = np.stack((poses_orb, poses_openvslam), axis=2)
target_input_traj = ground_truth

assert ground_truth.shape[0] == poses_orb.shape[0] == poses_openvslam.shape[0], "Mismatch in number of samples"

sequence_length = 10
num_sequences = train_input_traj.shape[0] - sequence_length
X = np.array([train_input_traj[i:i+sequence_length] for i in range(num_sequences)])
y = np.array([target_input_traj[i+sequence_length] for i in range(num_sequences)])

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)

model=Sequential([LSTM(128,return_sequences=False, input_shape=(sequence_length,7*2)), Dense(7)])
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

#random generator for now. Later this ll be the new results from both the SLAMs
test_traj_orb = np.random.rand(sequence_length, 7)
test_traj_openvslam = np.random.rand(sequence_length, 7)
test_traj = np.stack((test_traj_orb, test_traj_openvslam), axis=1).reshape(1, sequence_length, 7 * 2)

predicted_traj = model.predict(test_traj)
print(predicted_traj)
