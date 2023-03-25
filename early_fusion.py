import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import open3d as o3d
from yolov4.tf import YOLOv4
import tensorflow as tf
import time
import statistics
import random

from utils import *

if __name__ =="__main__":
    video_images, video_points, calib_files = data_path()

    lidar2cam_video = LiDAR2Camera(calib_files[0])

    yolo = yolo_make()

    for idx, img in enumerate(video_images):
        start_time = time.time()
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        img_final = lidar2cam_video.pipeline(image, point_cloud, yolo)
        exec_time = time.time() - start_time
        print("time: {:.2f} ms".format(exec_time * 1000))
        cv2.imshow("img_final", img_final)
        cv2.waitKey(1)
        # plt.imshow(img_final)
        # plt.show()
