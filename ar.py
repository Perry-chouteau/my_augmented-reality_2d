#!/usr/bin/env python3

import cv2
import sys
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

# v pick video or image
vid = cv2.VideoCapture("video.mp4")
#image_src = cv2.imread("image.jpg")
# ^ pick video or image

while cap.isOpened():
    success, imgWebcam = cap.read()

    # v using video
    success2, imgVid = vid.read()

    if not success2:
#        break
        vid = cv2.VideoCapture("video.mp4")
        success2, imgVid = vid.read()
        if not success2:
            break

    image_src = imgVid
    # ^ using video

    if not success:
        break

    scale_percent = 60 # percent of original size
    width = int(imgWebcam.shape[1] * scale_percent / 100)
    height = int(imgWebcam.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    imgWebcam = cv2.resize(imgWebcam, dim, interpolation = cv2.INTER_AREA)
    org_imageWebcam = imgWebcam.copy()

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejectedCandidates = detector.detectMarkers(imgWebcam)

    #imgWebcam = aruco.drawDetectedMarkers(imgWebcam, corners, ids)
    if  np.all(ids != None):
        for c in corners:
            #get corners (j'ai pas bien compris l'indexation ^^' mais c'est la doc)
            x1 = (c[0][0][0], c[0][0][1])
            x2 = (c[0][1][0], c[0][1][1])
            x3 = (c[0][2][0], c[0][2][1])
            x4 = (c[0][3][0], c[0][3][1])

            # calculer l'image et la surface
            id_distance = imgWebcam
            size = image_src.shape
            pts_dist = np.array([x1, x2, x3, x4])
            pts_src = np.array([[0, 0],
                                [size[1] - 1, 0],
                                [size[1] - 1, size[0] - 1],
                                [0, size[0] - 1]])
            # (homography) determine ensuite la transformation entre l'image et la surface
            h, status = cv2.findHomography(pts_src, pts_dist)
            temp = cv2.warpPerspective(image_src.copy(), h, (org_imageWebcam.shape[1], org_imageWebcam.shape[0]))
            cv2.fillConvexPoly(org_imageWebcam, pts_dist.astype(int), 0, 16)
            org_imageWebcam = cv2.add(org_imageWebcam, temp)
        cv2.imshow("imgWebcam", org_imageWebcam)
    else:
        cv2.imshow("imgWebcam", imgWebcam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        continue
    if cv2.getWindowProperty("imgWebcam", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()