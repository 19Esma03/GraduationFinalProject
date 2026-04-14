import cv2
import glob
import os
import numpy as np
import open3d as o3d
import re

from BackGround_Removal import remove_background
from Feature_Extraction import get_features, match_features, estimate_pose
from Depth_Estimation import get_depth_map

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def align_clouds_icp(source_pcd, target_pcd,initial_trans):
    threshold = 0.5
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return reg_p2p.transformation

INPUT_DIR = 'Input_Images'
OUTPUT_DIR = 'Output_Models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob.glob(f"{INPUT_DIR}/*.jpg") + glob.glob(f"{INPUT_DIR}/*.png"))
total_images = len(image_paths)

focal_length = 1200
K = np.array([[focal_length, 0, 640], [0, focal_length, 360], [0, 0, 1]], dtype=np.float64)

print(f"Total {total_images} images loaded. Extracting features...")

images_nobg = []
keypoints_list = []
descriptors_list = []

for path in image_paths:
    img = cv2.imread(path)
    img_nobg = remove_background(img)
    img_gray = cv2.cvtColor(img_nobg, cv2.COLOR_BGRA2GRAY)
    kp, des = get_features(img_gray)
    
    images_nobg.append(img_nobg)
    keypoints_list.append(kp)
    descriptors_list.append(des)

# BRUTE-FORCE MATRIX
print("Match Matrix creating")
match_scores = np.zeros(total_images)

for i in range(total_images):
    for j in range(total_images):
        if i != j:
            matches = match_features(descriptors_list[i], descriptors_list[j])
            match_scores[i] += len(matches) # Toplam eşleşme sayısını ekle

# En çok eşleşme bulan fotoğrafı Anchor yap
anchor_idx = np.argmax(match_scores)
print(f" Anchor Choosen: {os.path.basename(image_paths[anchor_idx])} (Skor: {match_scores[anchor_idx]})")

dense_global_points_3D = []

# Merkez kameranın konumunu sıfırla 
anchor_R = np.eye(3)
anchor_t = np.zeros((3, 1))

# Merkez kameranın kendi derinliğini ekle
anchor_depth = get_depth_map(images_nobg[anchor_idx])
alpha = images_nobg[anchor_idx][:, :, 3]
anchor_depth[alpha < 50] = 0 
h, w = anchor_depth.shape
intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, focal_length, focal_length, w/2, h/2)
anchor_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(anchor_depth.astype(np.float32)), intrinsic)
anchor_pcd = anchor_pcd.uniform_down_sample(every_k_points=10)
dense_global_points_3D.append(np.asarray(anchor_pcd.points))

for i in range(total_images):
    if i == anchor_idx: continue 
    
    matches = match_features(descriptors_list[anchor_idx], descriptors_list[i])
    
    if len(matches) > 30: # Daha katı bir sınır koyduk
        print(f"[{i+1}/{total_images}] {os.path.basename(image_paths[i])} Alligns to Center (Matches: {len(matches)})")
        
        R, t, _, _, mask = estimate_pose(keypoints_list[anchor_idx], keypoints_list[i], matches, K)
        
        # Derinlik Haritası
        depth_map = get_depth_map(images_nobg[i])
        alpha_channel = images_nobg[i][:, :, 3]
        depth_map[alpha_channel < 50] = 0 # Boşluk maskesi!
        
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
        pcd_local = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic)
        pcd_local = pcd_local.uniform_down_sample(every_k_points=10) # RAM koruması
        
        local_points = np.asarray(pcd_local.points)
        if len(local_points) > 0:
            
            global_dense = (R @ local_points.T).T + t.T
            dense_global_points_3D.append(global_dense)
    else:
        print(f"Warning: {os.path.basename(image_paths[i])} could not be aligned with the anchor camera and was skipped.")


if len(dense_global_points_3D) > 0:
    print("\n  Point clouds  merging.")
    all_dense_points = np.vstack(dense_global_points_3D)
    
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_dense_points)
    
    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    output_cloud_path = f"{OUTPUT_DIR}/anchor_dense_cloud.ply"
    o3d.io.write_point_cloud(output_cloud_path, final_pcd)
    print(f"Sparse point cloud saved to {output_cloud_path} ({len(all_dense_points)} points)")
    
    o3d.visualization.draw_geometries([final_pcd], window_name="Anchor Model Viewer", width=1280, height=720, point_show_normal=False)