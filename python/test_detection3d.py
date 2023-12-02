#!/usr/bin/env python3

import sys
import time
import cv2

import capnp
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import ecal.core.core as ecal_core
from capnp_subscriber import CapnpSubscriber

from ultralytics import YOLO
model = YOLO('models/yolov8n_gray.pt')

capnp.add_import_hook(['../src/capnp'])

import image_capnp as eCALImage
import disparity_capnp as eCALDisaprity
import odometry3d_capnp as eCALOdometry3d

first_message = True

class Detection3d():
    def __init__(self):
        self.target_coord = [] # Target Pixel coordinates in camera image
        self.target_depth = [] # Target depth in camera image
        self.target_3d_coord = [] # Target coordinates in camera frame

        self.target_position = [] # Target position in world frame
        self.target_position_old = [] # Prev Target position in world frame
        self.target_velocity = [] # Target velocity in world frame

        self.position = [] # Camera position in world frame
        self.velocity = [] # Camera velocity in world frame
        self.orientation = [] # Camera orientation in world frame

        self.image_map = {} # Gives annotated frame
        self.disparity_map = {} # Gives disparity map
        self.depth = None # Gives depth map

        self.camera_info = [] # [imageMsg.width, imageMsg.height, imageMsg.fx, imageMsg.fy, imageMsg.cx, imageMsg.cy, imageMsg.baseline]
        self.intrinsic = []

    def callback_image(self, type, topic_name, msg, ts):

        # need to remove the .decode() function within the Python API of ecal.core.subscriber StringSubscriber
        with eCALImage.Image.from_bytes(msg) as imageMsg:
            # print(f"seq = {imageMsg.header.seq}, stamp = {imageMsg.header.stamp}, with {len(msg)} bytes, encoding = {imageMsg.encoding}")
            # print(f"latency device = {imageMsg.header.latencyDevice / 1e6} ms")
            # print(f"latency host = {imageMsg.header.latencyHost / 1e6} ms")
            # print(f"width = {imageMsg.width}, height = {imageMsg.height}")
            # print(f"exposure = {imageMsg.exposureUSec}, gain = {imageMsg.gain}")
            # print(f"intrinsic = {imageMsg.intrinsic}")
            # print(f"extrinsic = {imageMsg.extrinsic}")

            if (imageMsg.encoding == "mono8"):

                mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
                mat = mat.reshape((imageMsg.height, imageMsg.width, 1))
                rgb_img = np.repeat(mat, 3, axis=-1)

                results = model.track(rgb_img, conf=0.3, iou=0.5, persist=True)

                for r in results:
                    if len(r.boxes.xywh) > 0:
                        self.target_coord = r.boxes.xywh[0].cpu().numpy()
                        # print(r.boxes.xywh[0])
                    else:
                        self.target_coord = []
                        
                annotated_frame = results[0].plot()
                self.image_map["detection"] = annotated_frame
                
            else:
                raise RuntimeError("Unused encoding: " + imageMsg.encoding)
            
    def callback_disparity(self, type, topic_name, msg, ts):
            # need to remove the .decode() function within the Python API of ecal.core.subscriber ByteSubscriber
        with eCALDisaprity.Disparity.from_bytes(msg) as imageMsg:

            self.camera_info = [imageMsg.width, imageMsg.height, imageMsg.fx, imageMsg.fy, imageMsg.cx, imageMsg.cy, imageMsg.baseline]
            self.intrinsic = np.array([[imageMsg.fx, 0, imageMsg.cx],[0, imageMsg.fy, imageMsg.cx],[0,0,1]])
            # print(f"seq = {imageMsg.header.seq}, with {len(msg)} bytes, encoding = {imageMsg.encoding}")
            # print(f"width = {imageMsg.width}, height = {imageMsg.height}")
            # print(f"[fx fy cx cy baseline] - sensor = [{imageMsg.fx} {imageMsg.fy} {imageMsg.cx} {imageMsg.cy} {imageMsg.baseline}] - {imageMsg.streamName}")

            if (imageMsg.encoding == "disparity16"):
                mat_uint16 = np.frombuffer(imageMsg.data, dtype=np.uint16)
                mat_uint16 = mat_uint16.reshape((imageMsg.height, imageMsg.width, 1))
                
                mat_float32 = mat_uint16.astype(np.float32) / 8.0
                mat_float32 = cv2.medianBlur(mat_float32, 5) # 5x5 median filter
                # mat_float32 = cv2.bilateralFilter(mat_float32, 9, 75, 75) # 9x9 bilateral filter
            
                disparity = (mat_float32 * (255.0 / imageMsg.maxDisparity)).astype(np.uint8)
                disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

                # Assuming imageMsg contains baseline and fx information
                max_depth = 5.0
                min_disparity_threshold = 1e-2  # Adjust this threshold based on your scene characteristics

                # Calculate depth, limiting the minimum disparity+
                epsilon = 1e-5
                depth = np.where(mat_float32 > min_disparity_threshold + epsilon,
                                (imageMsg.baseline * imageMsg.fx / (mat_float32 + epsilon)),
                                max_depth)

                # Limit the maximum depth to 5 meters
                depth = np.minimum(depth, max_depth)
                disp = (depth * (255.0 / 5.0)).astype(np.uint8)
                
                depth = cv2.resize(depth, (640, 416))  
                disparity = cv2.resize(disparity, (640, 416))             
                self.depth = depth
                self.disparity_map[topic_name + " disparity"] = disparity

    def callback_odom(self, type, topic_name, msg, time):

        global first_message

        # need to remove the .decode() function within the Python API of ecal.core.subscriber ByteSubscriber
        
        with eCALOdometry3d.Odometry3d.from_bytes(msg) as odometryMsg:
            # print(f"seq = {odometryMsg.header.seq}")
            # print(f"latency device = {odometryMsg.header.latencyDevice / 1e6} ms")
            # print(f"latency host = {odometryMsg.header.latencyHost / 1e6} ms")

            if first_message:
                print(f"bodyFrame = {odometryMsg.bodyFrame}")
                print(f"referenceFrame = {odometryMsg.referenceFrame}")
                print(f"velocityFrame = {odometryMsg.velocityFrame}")
                first_message = False

            # print(f"Camera position = {odometryMsg.pose.position.x}, {odometryMsg.pose.position.y}, {odometryMsg.pose.position.z}")
            # print(f"Camera orientation = {odometryMsg.pose.orientation.w}, {odometryMsg.pose.orientation.x}, {odometryMsg.pose.orientation.y}, {odometryMsg.pose.orientation.z}")

            self.position = [odometryMsg.pose.position.x, odometryMsg.pose.position.y, odometryMsg.pose.position.z]
            self.orientation = [odometryMsg.pose.orientation.w, odometryMsg.pose.orientation.x, odometryMsg.pose.orientation.y, odometryMsg.pose.orientation.z]
            # self.velocity = [odometryMsg.twist.linear.x, odometryMsg.twist.linear.y, odometryMsg.twist.linear.z]

    def get_target_position(self):

        # Gives position of Target in camera frame
        self.target_depth = self.depth[int(self.target_coord[1])][int(self.target_coord[0])]
        target_X = self.target_depth * ( self.target_coord[0] - self.camera_info[4]) / self.camera_info[2]
        target_Y = self.target_depth * ( self.target_coord[1] - self.camera_info[5]) / self.camera_info[3]
        target_Z = self.target_depth
        
        self.target_3d_coord = [target_X, target_Y, target_Z]

        r = R.from_quat(self.orientation)
        self.position = np.array(self.position).reshape((3, 1))

        # Create a 4x4 transformation matrix
        camera_to_world_matrix = np.eye(4)
        camera_to_world_matrix[:3, :3] = r.as_matrix()
        camera_to_world_matrix[:3, 3] = self.position.flatten()

        # camera_to_world_matrix = np.column_stack((r, self.position))
        # print(camera_to_world_matrix)
        # camera_to_world_matrix = np.vstack((camera_to_world_matrix, [0, 0, 0, 1])) # homogeneous transformation matrix from camera coordinates to world coordinates

        target_in_camera_coordinates = np.array([target_X, target_Y, target_Z, 1])
        self.target_position = np.dot(camera_to_world_matrix, target_in_camera_coordinates)[:3]

        # print(f"Target Coord = {target_in_camera_coordinates}")     
        # print(f"Target position = {self.target_position[0]}, {self.target_position[1]}, {self.target_position[2]}")     
        return self.target_position
    
    def get_target_velocity(self, curr, prev, dt): 
        # Gives target velocity in world frame
        self.target_velocity = (curr - prev) / (dt/1000)

def main():  

    # print eCAL version and date
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    
    # initialize eCAL API
    ecal_core.initialize(sys.argv, "test_image_sub")
    
    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    # create subscriber and connect callback
    topic1 = "S0/stereo1_l"
    topic2 = "S0/disparity/stereo1"
    topic3 = "S0/vio_odom"

    # n = len(sys.argv)
    # if n == 1:
    #     topic = "S0/stereo1_l"
    # elif n == 3:
    #     topic1 = sys.argv[1]

    # else:
    #     raise RuntimeError("Need to pass in exactly one parameter for topic")

    print(f"Streaming topic {topic1} and {topic2}")

    detection = Detection3d()
    sub_image = CapnpSubscriber("Image", topic1)
    sub_disparity = CapnpSubscriber("Disparity", topic2)
    sub_odom = CapnpSubscriber("Odometry3d", topic3)
    sub_image.set_callback(detection.callback_image)
    sub_disparity.set_callback(detection.callback_disparity)
    sub_odom.set_callback(detection.callback_odom)

    dt = 10 # ms
    prev_target_pos = [0, 0, 0]
        
    # idle main thread
    while ecal_core.ok():

        if detection.depth is not None and len(detection.target_coord) > 1:
            curr_target_pos = detection.get_target_position()
            current_target_vel = detection.get_target_velocity(curr_target_pos, prev_target_pos, dt)
            prev_target_pos = curr_target_pos
        
        for im in detection.image_map:
            if detection.depth is not None and len(detection.target_coord) > 1:
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"u: {detection.target_3d_coord[0]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"v: {detection.target_3d_coord[1]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"depth: {detection.target_3d_coord[2]:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"x: {detection.target_position[0]:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"y: {detection.target_position[1]:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"z: {detection.target_position[2]:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                detection.image_map[im] = cv2.putText(detection.image_map[im], f"vx: {detection.target_velocity[0]:.2f}", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"vy: {detection.target_velocity[1]:.2f}", (520, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                detection.image_map[im] = cv2.putText(detection.image_map[im], f"vz: {detection.target_velocity[2]:.2f}", (520, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(im, detection.image_map[im])

        for im in detection.disparity_map:
            if detection.depth is not None and len(detection.target_coord) > 1:
                centre = (int(detection.target_coord[0]), int(detection.target_coord[1]))
                cv2.circle(detection.disparity_map[im], centre, 5, (0,0,255),-1)
            # cv2.imshow(im, detection.disparity_map[im])
        cv2.waitKey(dt)
        # time.sleep(0.1)

    
    # finalize eCAL API
    ecal_core.finalize()

if __name__ == "__main__":
    main()
