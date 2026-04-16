"""
Feature_Extraction.py — Düzeltilmiş Versiyon
Değişiklikler:
  - Lowe oranı 0.60 → 0.75 (daha fazla ama daha güvenilir eşleşme)
  - estimate_pose_pnp eklendi (ölçekli pose için)
"""

import cv2
import numpy as np


def get_features(image_gray):
    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.02,
        edgeThreshold=15
    )
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        # 0.75 oranı: 0.60'dan daha permissive ama hâlâ güvenilir
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


def estimate_pose(keypoints1, keypoints2, matches, K):
    """
    Essential matrix üzerinden R, t tahmini.
    Dönen t ölçeksizdir — sadece yön bilgisi taşır.
    Gerçek ölçek için Main_Pipeline'daki estimate_scale_from_depth'i kullan.
    """
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None, pts1, pts2, None

    _, R, t, mask_rp = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, pts1, pts2, mask_rp


def estimate_pose_pnp(keypoints2d, points3d, K, dist_coeffs=None):
    """
    PnP ile ölçekli pose tahmini.
    keypoints2d : (N, 2) float array — görüntüdeki 2D noktalar
    points3d    : (N, 3) float array — dünya koordinatlarındaki 3D noktalar
    
    Bu fonksiyon loop closure veya gelişmiş pipeline için kullanılabilir.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4)

    if len(keypoints2d) < 6:
        return None, None, None

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d.astype(np.float64),
        keypoints2d.astype(np.float64),
        K,
        dist_coeffs,
        iterationsCount=1000,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    R_mat, _ = cv2.Rodrigues(rvec)
    return R_mat, tvec, inliers
