import cv2
import numpy as np
import os
import csv


#Camera Parameters for left & right
fx1=458.654
fy1=457.296
cx1=367.215
cy1=248.375

fx2=457.587
fy2=456.134
cx2=379.999
cy2=255.238

#TODO should this be fixed?
image_size = (752, 480)

# Example intrinsic and extrinsic parameters
K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])  # Camera 1 intrinsic matrix
K2 = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])  # Camera 2 intrinsic matrix

#Distortion Coefficient
D1 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05,0])
D2 = np.array([-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05,0])

#transformation matrix for each cam
cam_tran1=np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0]])

cam_tran2=np.array([[0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
        [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
        [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
        [0.0, 0.0, 0.0, 1.0]])

T_SB_left_inv = np.linalg.inv(cam_tran1)

T_right_left = np.dot(cam_tran2, T_SB_left_inv)
R=T_right_left[:3,:3]
T=T_right_left[:3,3]
B=np.linalg.norm(T)

def measure_blurriness(img_left,threshold=100):
    laplacian = cv2.Laplacian(img_left, cv2.CV_64F)
    laplacian_variance = laplacian.var()
    blur_bool=laplacian_variance<threshold
    return laplacian_variance,blur_bool

def calculate_entropy(img_left):    #texture indicator-higher entropy=more detailed texture
    hist = cv2.calcHist([img_left], [0], None, [256], [0, 256])
    # Normalize the histogram to get probabilities
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))  # Adding 1e-10 to avoid log(0)

    return entropy

def calculate_contrast(img_left,brightness):
    #Root Mean Square Contrast
    mean_intensity = np.mean(img_left)
    contrast = np.sqrt(np.mean((img_left - mean_intensity) ** 2))
    return contrast, brightness/contrast

def calculate_brightness(image):
    #Converting to grayscale in case of Colored input image
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(grayscale)
    return brightness

def calculate_disparity(img_left,img_right):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # Apply the rectification to the stereo images
    rectified_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)
    #MH blocksize=21
    #Vicon blocksize=17

    stereo = cv2.StereoBM_create(numDisparities=112, blockSize=21)

    #Calc the disparity map from the grayscale rectified images
    disparity = stereo.compute(rectified_left, rectified_right)
    disparity=disparity/16
    valid_disparity_mask = disparity > 0
    disparity[~valid_disparity_mask] = 0
    
    return disparity,np.mean(disparity),np.var(disparity),np.std(disparity),np.min(disparity),np.max(disparity)


def calculate_depth(disparity):
    '''
    This function calcs depth from Disparity values
    Formula: 
    Depth = (Baseline*Focal length) / Disparity
    '''
    depth = np.zeros_like(disparity)
    depth[disparity==0] = float("inf")  #infinite distance when there is no disparity
    
    depth = (((fx1 + fx2)/2)*B)/disparity

    return depth

def calculate_parameters(img_left,img_right):
    laplacian_variance,blur_bool=measure_blurriness(img_left)
    entropy=calculate_entropy(img_left)
    brightness = calculate_brightness(img_left)
    contrast, b_c_ratio=calculate_contrast(img_left, brightness)
    disparity,disparity_mean,disparity_var,disparity_std,disparity_min,disparity_max=calculate_disparity(img_left,img_right)
    #depth = calculate_depth(disparity)
    return [laplacian_variance,blur_bool,entropy,brightness,contrast,b_c_ratio,disparity_mean,disparity_var,disparity_std,disparity_min,disparity_max]

def main():
    # Folder path where images are stored
    folder_path = 'cam0/data'
    params=[0]
    # Generator function to yield image file paths one by one
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Initialize list to store rows (each row corresponds to an image)
    rows = [['timestamp','laplacian_variance','blur_bool','entropy','brightness','contrast', 'brightness_contrast_ratio','disparity_mean','disparity_var','disparity_std','disparity_min','disparity_max']]

    # Loop through each image, calculate parameters, and store them
    for filename in image_files:
        file_path_left = os.path.join('cam0/data', filename)
        file_path_right = os.path.join('cam1/data', filename)
        if not os.path.exists(file_path_right):
            print(f"File {file_path_right} does not exist. Skipping.")
            continue
        img_left = cv2.imread(file_path_left, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(file_path_right, cv2.IMREAD_GRAYSCALE)
        params[0] = filename[:-4]

        # Calculate parameters
        rows.append(params + calculate_parameters(img_left,img_right))
    
    # Save data to CSV file
    with open('image_feature.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Data saved")


if __name__ == "__main__":
    main()
