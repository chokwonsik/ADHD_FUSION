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
from sklearn.cluster import DBSCAN

from utils import *

if __name__ =="__main__":
    video_images, video_points, calib_files = data_path()

    lidar2cam_video = LiDAR2Camera(calib_files[0])

    yolo = yolo_make()

    #dbscan = DBSCAN(eps=0.5, min_samples=5)
    DOWN = 0.1

    for idx, img in enumerate(video_images):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        
        #down sampling
        down_cloud = o3d.geometry.PointCloud()
        down_cloud.points = o3d.utility.Vector3dVector(point_cloud)
        down_cloud = down_cloud.voxel_down_sample(voxel_size=DOWN)

        #Clustering
        # cluster_labels = dbscan.fit_predict(point_cloud[:,:3])        
        # n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        # cluster_points = [point_cloud[cluster_labels == i] for i in range(n_clusters)]
        # cluster_point_counts = [len(cluster) for cluster in cluster_points]  
        # print(f"Number of clusters: {n_clusters+1}")
        # print(f"Cluster point counts: {cluster_point_counts}")
        
        img_final = lidar2cam_video.pipeline(image, down_cloud, yolo)
        cv2.imshow("img_final", img_final)
        cv2.waitKey(3000)
