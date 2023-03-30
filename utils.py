import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
import glob
import open3d as o3d
from yolov4.tf import YOLOv4
import tensorflow as tf
import time
import statistics
import random

def data_path():
    # window
    video_images = sorted(glob.glob("data\\data_train\\img\\*.png"))
    video_points = sorted(glob.glob("data\\data_train\\velodyne_pcd\\*.pcd"))
    calib_files = sorted(glob.glob("data\\data_train\\calib\\*.txt"))
    
    # # linux
    # video_images = sorted(glob.glob("data/data_train/img/*.png"))
    # video_points = sorted(glob.glob("data/data_train/velodyne_pcd/*.pcd"))
    # calib_files = sorted(glob.glob("data/data_train/calib/*.txt"))
    # print(calib_files[0])
    return video_images, video_points, calib_files

def yolo_make():
    yolo = YOLOv4(tiny=True)
    yolo.classes = "data/coco.names"
    yolo.make_model()
    yolo.load_weights("data/yolov4-tiny.weights", weights_type="yolo")
    # yolo.load_weights("data/yolov4.weights", weights_type="yolo")
    return yolo

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R_rect"]
        self.R0 = np.reshape(R0, [3, 3])

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_image(self, pts_3d_velo):
        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        p_r0 = np.dot(self.P, R0_homo_2) #PxR0
        p_r0_rt =  np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1]))) #PxROxRT
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0],1))])
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))#PxROxRTxX
        pts_2d = np.transpose(p_r0_rt_x)
        
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def get_lidar_in_image_fov(self,pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
        pts_2d = self.project_velo_to_image(pc_velo)
        fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo
    
    def lidar_camera_fusion(self, pred_bboxes, image, pc_velo):
        img_bis = image.copy()
        
        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img_bis.shape[1], img_bis.shape[0], True
        )
        self.imgfov_pts_2d = pts_2d[fov_inds, :]
        self.imgfov_pc_velo = imgfov_pc_velo

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for box in pred_bboxes:
            distances = []
            for i in range(self.imgfov_pts_2d.shape[0]):
                depth = self.imgfov_pc_velo[i,0]
                if (rectContains(box, self.imgfov_pts_2d[i], image.shape[1], image.shape[0], shrink_factor=0.25)==True):
                    distances.append(depth)

                    color = cmap[int(510.0 / depth), :]
                    cv2.circle(img_bis,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),2,color=tuple(color),thickness=-1,)
            h, w, _ = img_bis.shape
            if (len(distances)>2): # 2 meter
                distances = filter_outliers(distances)
                best_distance = get_best_distance(distances, technique="closet")
                cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0]*w),int(box[1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)    
            distances_to_keep = []
        # img_bis = cv2.cvtColor(img_bis, cv2.COLOR_BGR2RGB)
        
        return img_bis, distances
    
    def pipeline(self, image, point_cloud, yolo):
        img = image.copy()
        
        #Code for downsampling (cuda error)
        point_cloud_array = np.asarray(point_cloud.points)

        # start_time=time.time()
        result, pred_bboxes = run_obstacle_detection(img, yolo)
        img_final, _ = self.lidar_camera_fusion(pred_bboxes, result, point_cloud_array[:,:3])
        # exec_time = time.time() - start_time
        # print("time: {:.2f} ms".format(exec_time * 1000))

        return img_final, pred_bboxes


def filter_outliers(distances):
    inliers = []
    mu  = statistics.mean(distances)
    std = statistics.stdev(distances)
    for x in distances:
        if abs(x-mu) < std:
            # This is an INLIER
            inliers.append(x)
    return inliers

def get_best_distance(distances, technique="closest"):
    if technique == "closest":
        return min(distances)
    elif technique =="average":
        return statistics.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return statistics.median(sorted(distances))
    
def rectContains(rect,pt, w, h, shrink_factor = 0):       
    x1 = int(rect[0]*w - rect[2]*w*0.5*(1-shrink_factor)) # center_x - width /2 * shrink_factor
    y1 = int(rect[1]*h-rect[3]*h*0.5*(1-shrink_factor)) # center_y - height /2 * shrink_factor
    x2 = int(rect[0]*w + rect[2]*w*0.5*(1-shrink_factor)) # center_x + width/2 * shrink_factor
    y2 = int(rect[1]*h+rect[3]*h*0.5*(1-shrink_factor)) # center_y + height/2 * shrink_factor
    
    return x1 < pt[0]<x2 and y1 <pt[1]<y2

def run_obstacle_detection(img, yolo):
    # start_time=time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = yolo.resize_image(img)
    # 0 ~ 255 to 0.0 ~ 1.0
    resized_image = resized_image / 255.
    #input_data == Dim(1, input_size, input_size, channels)
    input_data = resized_image[np.newaxis, ...].astype(np.float32)

    candidates = yolo.model.predict(input_data)

    _candidates = []
    result = img.copy()
    for candidate in candidates:
        batch_size = candidate.shape[0]
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
        candidates = np.concatenate(_candidates, axis=1)
        # pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.50, score_threshold=0.40)

        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)]
        pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        
        car_list = []
        
        for idx in range(len(pred_bboxes)):
            if int(pred_bboxes[:, 4][idx]) == 2:
                car_list.append(pred_bboxes[idx])
            elif int(pred_bboxes[:, 4][idx]) == 5:
                pred_bboxes[:, 4][idx] = 2
                car_list.append(pred_bboxes[idx])
                pass
            elif int(pred_bboxes[:, 4][idx]) == 7:
                pred_bboxes[:, 4][idx] = 2
                car_list.append(pred_bboxes[idx])
                pass
            else:
                continue

        car_list = np.array(car_list)
        
        try:
            result = yolo.draw_bboxes(img, car_list)
            # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        except:
            pass
    # exec_time = time.time() - start_time
    # print("time: {:.2f} ms".format(exec_time * 1000))
    return result, car_list
