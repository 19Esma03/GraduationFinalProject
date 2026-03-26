import cv2
import numpy as np
import open3d as o3d

img1 = cv2.imread("Image1.jpg")
img2 = cv2.imread("Image2.jpg")

#Gri Tona Cevir analiz kolay olsun
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#ORB la 5000 belirgin nokta bul 
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(gray1,None)#(kp= anahtar noktalar, des= tanımlayıcılar)
kp2, des2 = orb.detectAndCompute(gray2,None)

#BFMatcher: iki resmin ortak noktalarını karşılaştır
bf= cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches,key = lambda x: x.distance) #3 En iyi eşleşmeleri başa al

#Eşleşen noktaların (x,y) kordinatlarını listelere al 
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

#F = fundemental matrix (iki görüntü arasındaki geometrik ilişkiyi kurar)
#RANSAC = Hatalı eşleşmeleri temizlemek için kullanılır
F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)

#Sadece Doğru(maskelenmiş noktlaarı tut)
pts1=pts1[mask.ravel()==1]
pts2=pts2[mask.ravel()==1]


#K = Kamera Matrisi (kameranın odak uzaklığını ve merkez noktasını içerir)
h,w =gray1.shape
focal_length = 0.8 * w #Yaklaşık odak uzaklığı
K=np.array([[focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]])

#Essential Matix(E) kameranın dış parametrelerini(rotasyon ve çeviri)
E = K.T @ F @ K

#recoverPose: kameranın ne kadar döndüğünü(R) ve ne kadar kaydığını (t) hesaplar
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

#projeksiyon matrisleri:Kameranın dünya kordinatlarını belirler
proj1 = K @ np.hstack((np.eye(3), np.zeros((3,1)))) #İlk kamera başlangıç noktasında
proj2 = K @ np.hstack((R, t)) #İkinci kamera R ve t kadar uzakta

#TriangulatePoints: 2D noktaları çakıştırarak 4D homojen koordinatlar üretir
points_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
#Homojen koordinatları 3D koordinatlara çevir
points_3d = points_4d [:3]/ points_4d[3] 

#Open3D ile 3D noktaları görselleştir
points = points_3d.T

pcd = o3d.geometry.PointCloud()#Boş bir nokta bulutu objesi oluştur
pcd.points = o3d.utility.Vector3dVector(points) #Hesaplanan noktaları içine yükle

o3d.visualization.draw_geometries([pcd]) #Nokta bulutunu görselleştir

