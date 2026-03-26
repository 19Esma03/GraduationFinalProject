import cv2
import matplotlib.pyplot as plt
import numpy as np
from rembg import remove
from transformers import pipeline
from PIL import Image



img1_color = cv2.imread('foto1.jpg')
img2_color = cv2.imread('foto2.jpg')
#Islemi hizlandirmak icin boyut kucult
img1_color = cv2.resize(img1_color, (0,0), fx=0.5, fy=0.5)
img2_color = cv2.resize(img2_color, (0,0), fx=0.5, fy=0.5)

print("Arka planlar siliniyor, lütfen bekle...")
img1_nobg = remove(img1_color)
img2_nobg = remove(img2_color)

img1 = cv2.cvtColor(img1_nobg, cv2.COLOR_BGRA2GRAY)
img2 = cv2.cvtColor(img2_nobg, cv2.COLOR_BGRA2GRAY)

img_rgb = cv2.cvtColor(img1_nobg, cv2.COLOR_BGRA2RGB)
pil_image = Image.fromarray(img_rgb)

print("Yapay Zeka Modeli Yükleniyor")
depth_estimator = pipeline(task= "depth-estimation", model = "Intel/dpt-large")

if img1 is None or img2 is None:
    print("Hata: Gorseller bulunamadi. Lutfen dosya yollarini kontrol et.")
    exit()

#Scale-Invariant Feature Transform nesnesini başlat
sift = cv2.SIFT_create()
# Keypoints ve Descriptors  çıkarımı yap
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# knnMatch ile en yakın 2 noktayı bul ve Lowe's Ratio Test uygula
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
#Eslesen nokta kordinatları 
pts1 = np.float32([keypoints_1[m[0].queryIdx].pt for m in good_matches])
pts2 = np.float32([keypoints_2[m[0].trainIdx].pt for m in good_matches])

# RANSAC, hatalı eşleşmeleri dışarıda bırakan bir "maske" oluşturur
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
matchesMask = mask.ravel().tolist()

# Sadece RANSAC'tan geçen noktaları yeşil renkle çiz
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
matched_img_ransac = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, [m[0] for m in good_matches], None, **draw_params)

#Sadece güvenilir (good) eşleşmeleri görselleştir

#matched_img = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 5))
plt.imshow(matched_img_ransac)
plt.title("Maskelenmiş Temiz Eşleşme")
plt.show()

pts1_clean = np.float32([pts1[i] for i in range(len(matchesMask)) if matchesMask[i] == 1])
pts2_clean = np.float32([pts2[i] for i in range(len(matchesMask)) if matchesMask[i] == 1])

# Fotoğraf boyutundan tahmini bir odak uzaklığı kullanıyoruz.
h, w = img1.shape
focal_length = w 
center = (w // 2, h // 2)
K = np.array([[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]], dtype=np.float64)

#  Essential Matrix (Temel Matris) Hesaplama
E, mask_E = cv2.findEssentialMat(pts1_clean, pts2_clean, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

#  Kamera Pozisyonunu Çıkarma (R: Rotasyon Matrisi, t: Çeviri Vektörü)
_, R, t, mask_pose = cv2.recoverPose(E, pts1_clean, pts2_clean, K)

# Projeksiyon Matrislerini Oluşturma
Proj1 = np.hstack((np.eye(3), np.zeros((3, 1)))) # merkezde (0,0,0)
Proj2 = np.hstack((R, t)) #  R  t kadar hareket etmiş

Proj1 = K @ Proj1
Proj2 = K @ Proj2

points_4d_hom = cv2.triangulatePoints(Proj1, Proj2, pts1_clean.T, pts2_clean.T)

#  3D (X, Y, Z) koordinatlarına çevir
points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
points_3d = points_3d.T

# 3D Nokta Bulutu
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Noktaları çiz 
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='.', s=15)

ax.set_title("Seyrek 3D Nokta Bulutu (Sparse Point Cloud)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Eksen şit oranlı
ax.set_box_aspect([1,1,1]) 

plt.show()

h, w = img1.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_clean, pts2_clean, F, imgSize=(w, h))

# Görüntüleri yeni perspektife göre bük (Warp)
img1_rectified = cv2.warpPerspective(img1, H1, (w, h))
img2_rectified = cv2.warpPerspective(img2, H2, (w, h))

# #  Derinlik Haritası (Disparity Map) Hesaplama - StereoSGBM Algoritması
# window_size = 5
# min_disp = 0
# num_disp = 16 * 5 # Her zaman 16'nın katı olmak zorundadır

# stereo = cv2.StereoSGBM_create(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     blockSize=window_size,
#     P1=8 * 3 * window_size ** 2,
#     P2=32 * 3 * window_size ** 2,
#     disp12MaxDiff=1,
#     uniquenessRatio=10,
#     speckleWindowSize=100,
#     speckleRange=32
# )

# print("Derinlik haritasi hesaplaniyor, bu işlem biraz sürebilir...")
# # Derinliği hesapla (OpenCV formatı gereği 16'ya bölüyoruz)
# disparity = stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0

# #0-255 arasına normalize et
# disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # 9. Sonucu Görselleştir
# plt.figure(figsize=(10, 5))
# plt.imshow(disp_vis, cmap='jet') # Renkli ısı haritası görünümü için 'jet' kullanıyoruz
# plt.title("Derinlik Haritasi (Disparity Map) - Rektifiye Edilmiş")
# plt.colorbar(label="Derinlik (Yakin = Kırmızı/Sarı, Uzak = Mavi)")
# plt.show()


print("Derinlik Haritasi Tahmin Ediliyor")
predictions = depth_estimator(pil_image)

depth_image = predictions["depth"]
depth_map = np.array(depth_image)

plt.figure(figsize=(10,))
plt.imshow(depth_map, cmap='inferno') 
plt.colorbar(label="Göreceli Derinlik (Sarı/Beyaz = Yakın, Siyah/Mor = Uzak)")
plt.title("Yapay Zeka (MiDaS) ile Kusursuz Derinlik Haritası")
plt.axis('off')
plt.show()

