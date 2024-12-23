import rosbag
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
from std_msgs.msg import Header

bridge = CvBridge()

def create_image_message(image_path, timestamp):
    cv_image = cv2.imread(image_path)
    header = Header()
    header.stamp = rospy.Time.from_sec(timestamp)  
    image_message = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    image_message.header = header
    return image_message

def extract_timestamp_from_filename(filename):
    
    timestamp_str = filename.split('.')[0]  
    timestamp = float(timestamp_str) / 1e9  # Convert nanoseconds to seconds
    return timestamp

def create_rosbag(images_path, output_bag_file):
    with rosbag.Bag(output_bag_file, 'w') as bag:
        i=0
        for image_path in images_path:
            print(i)
            filename = os.path.basename(image_path)
            timestamp = extract_timestamp_from_filename(filename)
            image_msg = create_image_message(image_path, timestamp)
            bag.write('/camera/image_raw', image_msg, image_msg.header.stamp)
            i+=1

if __name__ == "__main__":
    images_path = sorted([os.path.join("data/", f) for f in os.listdir("data/") if f.endswith('.bmp')])
    output_bag_file = 'kagaru.bag'
    create_rosbag(images_path, output_bag_file)

    print(f"Rosbag {output_bag_file} created successfully!")
