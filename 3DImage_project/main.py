import cv2
import glob
import os
import numpy as np
import open3d as o3d
import re
from PIL import Image
from PIL.ExifTags import TAGS

from BackGround_Removal import remove_background
from Feature_Extraction import get_features, match_features, estimate_pose
from Depth_Estimation import get_depth_map

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def get_local_from_exit(image_path, sensor_width_mm = None):

    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None
        tags = {TAGS.get(k,k):v for k, v in exif_data.items()}
        fl_mm = tags.get('FocalLength')
        if fl_mm is None:
            return None
        fl_mm = float(fl_mm)

        #Sesör genişliği biliniyorsa piksele çevir
        if sensor_width_mm:
            img_w, _ = img.size
            focal_px = fl_mm * img_w / sensor_width_mm
            return focal_px
        
         # FocalLengthIn35mmFilm varsa 35mm eşdeğerinden tahmin et
        fl35 = tags.get("FocalLengthIn35mmFilm")
        if fl35:
            img_w, _ =img.size

            focal_px = fl35 * img_w / 36
            return focal_px
        
        return None

    except Exception:
        return None


def normalize_depth_map(depth_map, target_median = 3.0):

    valid = depth_map[depth_map > 0]
    if len(valid)==0:
        return depth_map
    current_median = np.median(valid)
    if current_median < 1e-6:
        return depth_map
    scale = target_median / current_median
    return depth_map * scale


def build_point_cloud(depth_map_raw, alpha_mask, intrinsic, depth_median = 3.0,downsaple_k=5, alpha_treshold = 50):

    #Tek Görüntüden nokta bulutu üretir

    depth_map = normalize_depth_map(depth_map_raw.copy(), target_median=depth_median)
    depth_map[alpha_mask < alpha_treshold] = 0

    #Kenar Gürültüsünü azalt: Alpha maskesine erozyon uygula
    kernel = np.ones((5,5), np.uint8)
    alpha_eroded = cv2.erode(alpha_mask, kernel, iterations=2)
    depth_map[alpha_eroded < alpha_treshold] = 0

    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic)

    if len(pcd.points) == 0:
        return pcd
    
    pcd = pcd.uniform_down_sample(every_k_points=downsaple_k)

    #Anlık outlier temizleme
    if len(pcd.points) > 50:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

    return pcd

def estimate_scale_from_depth(pts2d_prev, pts2d_curr, depth_prev,depth_curr, K):
    """
    2D eşleşme noktaları ve derinlik haritalarını kullanarak
    t vektörü için gerçek ölçeği tahmin eder.
    recoverPose'un ölçeksiz t'sini düzeltir.
    """
    scales = []

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth_prev.shape

    
    for (u1,v1),(u2,v2) in zip(pts2d_prev, pts2d_curr):
        u1,v1 = int(round(u1)), int(round(v1))
        u2,v2 = int(round(u2)), int(round(v2))

        if not (0<= v1 <h and 0 <= u1<w):
                continue
        if not (0<= v2 <h and 0 <= u2<w):
                continue
        d1 = depth_prev[v1,u1]
        d2 = depth_curr[v2,u2]
        if d1 <= 0 or d2 <= 0:
            continue
        #3D nokta tahmini(önceki)
        x1 = (u1 - cx) * d1 / fx
        y1 = (v1 - cy) * d1 / fy
        z1 = d1
        #3D nokta tahmini(sonraki)
        x2 = (u2 - cx) * d2 / fx
        y2 = (v2 - cy) * d2 / fy
        z2 = d2
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        scales.append(dist)

    if (len(scales) < 5):
        return 1.0

    return float(np.median(scales))

def allign_clouds_icp(source_pcd, target_pcd, initial_trans, treshold = 0.3):
    """
    İki nokta bulutunu ICP ile hizalar.
    source: Hareket ettirilecek nokta bulutu
    target: Referans nokta bulutu
    max_correspondence_distance: Eşleşme için maksimum mesafe
    init: Başlangıç dönüşümü (4x4 matrisi)
    """
    reg = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, initial_trans,
        o3d.pipelines.registration.TransfotmationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    if reg.fitness < 0.3:
        print(f"ICP düşük fitness: {reg.fitness:.4f}")
        return initial_trans
    print(f"Icp fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
    return reg.transformation


INPUT_DIR    = 'Input_Images'
OUTPUT_DIR   = 'Output_Models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Derinlik ölçek hedefi (objenin yaklaşık mesafesi, metre)
DEPTH_TARGET_MEDIAN =3.0
DOWNSAMPLE_K = 5
# ICP eşiği
ICP_TRESHOLD = 0.3
MIN_MATCHES = 30

image_paths = sorted( 
    glob.glob(f"[INPUT_DIR]/*.jpg"), glob.glob(f"{INPUT_DIR}/*.png"),
    key=natural_sort_key)

total_images = len(image_paths)
print(f"Toplam {total_images} image upload.")

img_test = cv2.imread(image_paths[0])
h_img, w_img = img_test.shape[:2]

focal_px = get_local_from_exit(image_paths[0])
if focal_px:
    print(f"Focal length from EXIF: {focal_px:.2f} px")
else:
    focal_px = 0.9 * w_img 
    print(f"EXIF'ten focal length bulunamadı, varsayılan: {focal_px:.2f} px")

K = np.array([
    [focal_px, 0, w_img/2],
    [0, focal_px, h_img/2],
    [0, 0, 1]
], dtype=np.float64)

intrinsic = o3d.camera.PinholeCameraIntrinsic(w_img, h_img, focal_px, focal_px, w_img/2, h_img/2)

print(f"\n[1/{total_images}] İlk kare işleniyor: {os.path.basename(image_paths[0])}")

img_prev      = cv2.imread(image_paths[0])
nobg_prev     = remove_background(img_prev)
gray_prev     = cv2.cvtColor(nobg_prev, cv2.COLOR_BGRA2GRAY)
alpha_prev    = nobg_prev[:, :, 3]
kp_prev, des_prev = get_features(gray_prev)
depth_raw_prev = get_depth_map(nobg_prev)
depth_norm_prev = normalize_depth_map(depth_raw_prev.copy(), DEPTH_TARGET_MEDIAN)

pcd_prev_original = build_point_cloud(
    depth_raw_prev, alpha_prev, intrinsic,
    depth_median=DEPTH_TARGET_MEDIAN,
    downsample_k=DOWNSAMPLE_K
)

global_pcd      = o3d.geometry.PointCloud()
global_pcd     += pcd_prev_original
cumulative_pose = np.eye(4)   # global koordinat sistemindeki poz

skipped = 0

for i in range(1, total_images):
    print(f"\n[{i+1}/{total_images}] {os.path.basename(image_paths[i])}")

    img_curr   = cv2.imread(image_paths[i])
    nobg_curr  = remove_background(img_curr)
    gray_curr  = cv2.cvtColor(nobg_curr, cv2.COLOR_BGRA2GRAY)
    alpha_curr = nobg_curr[:, :, 3]
    kp_curr, des_curr = get_features(gray_curr)

    from Feature_Extraction import match_features
    matches = match_features(des_prev, des_curr)
    print(f"  Eşleşme sayısı: {len(matches)}")

    if len(matches) < MIN_MATCHES:
        print(f"  [Atlandı] Yeterli eşleşme yok ({len(matches)} < {MIN_MATCHES})")
        skipped += 1
        continue

    # --- Pose tahmini ---
    pts1 = np.float32([kp_prev[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp_curr[m.trainIdx].pt for m in matches])

    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print("  [Atlandı] Essential matrix bulunamadı")
        skipped += 1
        continue

    _, R, t_unit, mask_rp = cv2.recoverPose(E, pts1, pts2, K)

    # --- Derinlik haritası ---
    depth_raw_curr  = get_depth_map(nobg_curr)
    depth_norm_curr = normalize_depth_map(depth_raw_curr.copy(), DEPTH_TARGET_MEDIAN)

    # --- Ölçek tahmini ---
    inlier_mask = mask_rp.ravel() > 0
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    scale = estimate_scale_from_depth(
        pts1_in, pts2_in, depth_norm_prev, depth_norm_curr, K
    )
    t_scaled = t_unit * scale
    print(f"  Tahmin edilen t ölçeği: {scale:.4f}")

    # --- Göreceli transform (ardışık kareler arası) ---
    T_rel = np.eye(4)
    T_rel[:3, :3] = R
    T_rel[:3,  3] = t_scaled.flatten()

    # --- Nokta bulutu oluştur (orijinal, transform edilmemiş) ---
    pcd_curr_original = build_point_cloud(
        depth_raw_curr, alpha_curr, intrinsic,
        depth_median=DEPTH_TARGET_MEDIAN,
        downsample_k=DOWNSAMPLE_K
    )

    if len(pcd_curr_original.points) < 50:
        print("  [Atlandı] Yeterli nokta üretilemedi")
        skipped += 1
        continue

    # --- ICP iyileştirme ---
    print("  ICP çalışıyor...")
    T_icp = align_clouds_icp(
        pcd_curr_original, pcd_prev_original, T_rel, threshold=ICP_THRESHOLD
    )

    # --- Kümülatif global poz ---
    cumulative_pose = cumulative_pose @ T_icp

    # --- Global sisteme dönüştür ve ekle ---
    pcd_global = o3d.geometry.PointCloud(pcd_curr_original)
    pcd_global.transform(cumulative_pose)
    global_pcd += pcd_global

    # --- Bir sonraki iterasyon için referanslar ---
    # KRİTİK: transform EDİLMEMİŞ orijinal bulutu referans olarak sakla
    kp_prev, des_prev     = kp_curr, des_curr
    depth_raw_prev        = depth_raw_curr
    depth_norm_prev       = depth_norm_curr
    pcd_prev_original     = pcd_curr_original   # dönüştürülmemiş!


print(f"\n{'='*50}")
print(f"İşlem tamamlandı. Atlanan kare: {skipped}/{total_images}")

if global_pcd.is_empty():
    print("HATA: Nokta bulutu boş!")
else:
    print(f"Ham nokta sayısı: {len(global_pcd.points)}")

    # Voxel downsample (uniform yoğunluk)
    voxel_size = 0.02
    global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Voxel sonrası nokta sayısı: {len(global_pcd.points)}")

    # İstatistiksel outlier temizleme
    cl, ind = global_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    global_pcd = global_pcd.select_by_index(ind)
    print(f"Temizleme sonrası nokta sayısı: {len(global_pcd.points)}")

    # Normaller hesapla (mesh oluşturma için gerekli)
    global_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    global_pcd.orient_normals_consistent_tangent_plane(100)

    # Kaydet
    out_path = f"{OUTPUT_DIR}/fixed_cloud.ply"
    o3d.io.write_point_cloud(out_path, global_pcd)
    print(f"Kaydedildi: {out_path}")

    # Görüntüle
    o3d.visualization.draw_geometries(
        [global_pcd],
        window_name="Düzeltilmiş Nokta Bulutu",
        width=1280, height=720,
        point_show_normal=False
    )

