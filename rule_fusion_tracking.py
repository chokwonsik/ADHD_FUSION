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
import argparse

from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

from utils import *

if __name__ =="__main__":
        
    tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)

    video_images, video_points, calib_files = data_path()

    lidar2cam_video = LiDAR2Camera(calib_files[0])

    yolo = yolo_make()

    result_lidar_video = []
    
    for idx, img in enumerate(video_images):
        # start_time=time.time()
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        point_cloud = o3d.io.read_point_cloud(video_points[idx])
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)
        _, inliers = point_cloud.segment_plane(distance_threshold = 0.25, ransac_n = 8, num_iterations = 500)

        
        point_cloud.points = o3d.utility.Vector3dVector(np.delete(np.asarray(point_cloud.points), inliers, axis=0))
        
        img_final, pred_bboxes = lidar2cam_video.pipeline(image, point_cloud, yolo)
        # print(pred_bboxes)
        if len(pred_bboxes) == 0:
            # print("detect X")
            cv2.imshow("img_final", img_final)
        else:
            arr4 = np.stack((np.array(pred_bboxes[:,0] * 1242), np.array(pred_bboxes[:,1] * 375), 
                             np.array(pred_bboxes[:,2] * 1242), np.array(pred_bboxes[:,3] * 375)), axis=1)
            tracks = tracker.update(arr4, pred_bboxes[:,5], np.array(pred_bboxes[:,4], np.int8))
            img_final = draw_tracks(img_final, tracks)
            print(tracks)
        result_lidar_video.append(img_final)
        cv2.imshow("img_final", img_final)
        cv2.waitKey(1)
        # exec_time = time.time() - start_time
        # print("time: {:.2f} ms".format(exec_time * 1000))



    out = cv2.VideoWriter('output/resulte.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (image.shape[1],image.shape[0]))
    for i in range(len(result_lidar_video)):
        out.write(result_lidar_video[i])
    out.release()
