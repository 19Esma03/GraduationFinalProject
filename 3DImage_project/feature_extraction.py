import cv2
import numpy as np

def get_features(image_gray):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_pose(keypoints1,keypoints2,matches,K):
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1,pts2,K,method = cv2.RANSAC,prob = 0.999,threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2,K)
    return R, t,pts1,pts2,mask