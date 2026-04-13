import cv2
import glob
import os
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

from BackGround_Removal import remove_background
from Feature_Extraction import get_features, match_features, estimate_pose
from Depth_Estimation import get_depth_map

INPUT_DIR = 'Input_Images'
OUTPUT_DIR = 'Output_Models'

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob.glob(f"{INPUT_DIR}/*.jpg")
total_images = len(image_paths)
focal_length = 1200
K = np.array([[focal_length, 0, 640],[0, focal_length, 360],[0, 0, 1]], dtype=np.float64)
global_points_3D = [];
dense_global_points_3D = []

prev_R = np.eye(3)
prev_t = np.zeros((3, 1))
current_R = np.eye(3)
current_t = np.zeros((3, 1))

# if total_images < 25:
#     print("Uyarı: 25'ten az görsel bulundu. Daha fazla görsel ekleyerek daha iyi sonuçlar elde edebilirsiniz.")


# else:
#     print("Görseller yeterli sayıda bulundu. İşlem başlatılıyor...")
    
print(f"Toplam {total_images} görsel bulundu. İşleniyor...")
img_prev = cv2.imread(image_paths[0])
img_prev_nobg = remove_background(img_prev)
img_prev_gray = cv2.cvtColor(remove_background(img_prev), cv2.COLOR_BGRA2GRAY)
kp_prev, des_prev = get_features(img_prev_gray)

depth_map_prev = get_depth_map(img_prev_nobg)
h,w = depth_map_prev.shape
intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, focal_length, focal_length, w/2, h/2)
depth_o3d_prev = o3d.geometry.Image(depth_map_prev.astype(np.float32))
pcd_prev = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d_prev, intrinsic)
dense_global_points_3D.append(np.asarray(pcd_prev.points))

for i in range(1, len(image_paths)):

    img_curr = cv2.imread(image_paths[i])
    img_curr_nobg = remove_background(img_curr)
    img_curr_gray = cv2.cvtColor(img_curr_nobg, cv2.COLOR_BGRA2GRAY)

    kp_curr, des_curr = get_features(img_curr_gray)
    matches = match_features(des_prev, des_curr)

    if len(matches) > 10:
        R,t,pts1,pts2,mask = estimate_pose(kp_prev, kp_curr, matches, K)
        current_t = current_t + current_R @ t
        current_R = R @ current_R

        Proj1 = K @ np.hstack((prev_R, prev_t))
        Proj2 = K @ np.hstack((current_R, current_t))

        points_4D = cv2.triangulatePoints(Proj1, Proj2, pts1.T, pts2.T)
        points_3D = points_4D / points_4D[3, :]
        global_points_3D.append(points_3D[:3, :].T)

        depth_map = get_depth_map(img_curr_nobg)
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

        pcd_local = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic)
        local_points = np.asarray(pcd_local.points)

        if len(local_points) > 0:
            # Yerel noktaları mevcut kamera açısına göre Global Kordinata taşı
            global_dense = (current_R @ local_points.T).T + current_t.T
            dense_global_points_3D.append(global_dense)


    kp_prev, des_prev = kp_curr, des_curr
    prev_R = current_R.copy()
    prev_t = current_t.copy()

else:
        print(f"ERROR: {os.path.basename(image_paths[i])} Image doesn't resemble the previous .")

if len(dense_global_points_3D) > 0:

    all_dense_points = np.vstack(dense_global_points_3D)

    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_dense_points)
    
    output_cloud_path = f"{OUTPUT_DIR}/dense_cloud.ply"
    o3d.io.write_point_cloud(output_cloud_path, final_pcd)
    print(f"Dense Sparse Cloud Saved: {output_cloud_path} ({len(all_dense_points)} nokta)")


