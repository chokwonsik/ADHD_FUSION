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
    
    # time.sleep(10)

    result_lidar_video = []
    
    for idx, img in enumerate(video_images):
        # start_time=time.time()
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        point_cloud = o3d.io.read_point_cloud(video_points[idx])
        # downsampling
        point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)
        # remove ground
        _, inliers = point_cloud.segment_plane(distance_threshold = 0.25, ransac_n = 8, num_iterations = 500)
        point_cloud.points = o3d.utility.Vector3dVector(np.delete(np.asarray(point_cloud.points), inliers, axis=0))
        # remove large_cluster
        # try:
        #     labels = np.array(point_cloud.cluster_dbscan(eps=0.2, min_points=5, print_progress=True))
        #     point_cloud = remove_large_clusters(point_cloud, labels, 4000)
            
        #     max_label = labels.max()

        #     # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        #     # colors[labels < 0] = 0
        #     # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        #     point_cloud = point_cloud.select_by_index(np.where(labels >= 0)[0])
        #     # time.sleep(0.1)
        # except:
        #     pass
        
        img_final, pred_bboxes, result_yolo = lidar2cam_video.pipeline(image, point_cloud, yolo)
        cv2.imshow("img_final_yolo", result_yolo)
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
        result_lidar_video.append(result_yolo)
        cv2.imshow("img_final", img_final)
        if cv2.waitKey(1) == ord('q'):
            break
        # exec_time = time.time() - start_time
        # print("time: {:.2f} ms".format(exec_time * 1000))



    out = cv2.VideoWriter('output/result_yolo.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (image.shape[1],image.shape[0]))
    for i in range(len(result_lidar_video)):
        out.write(result_lidar_video[i])
    out.release()
