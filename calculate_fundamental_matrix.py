import numpy as np
import cv2

print(cv2.__version__)

frames = {}
camera_combinations = [("0","1"),("0","2"),("0","3"),("1","2"),("1","3"),("2","3")]

cap = cv2.VideoCapture("./p001g15/p001g15c00.mp4")
ret, frame = cap.read()
frames["0"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c01.mp4")
ret, frame = cap.read()
frames["1"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c02.mp4")
ret, frame = cap.read()
frames["2"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c03.mp4")
ret, frame = cap.read()
frames["3"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

for camera_combination in camera_combinations:

    camLeft = camera_combination[0]
    camRight = camera_combination[1]

    imgLeft = frames[camLeft]
    imgRight = frames[camRight]

    sift = cv2.xfeatures2d.SIFT_create()
    keyPointsLeft, descriptorsLeft = sift.detectAndCompute(imgLeft, None)
    keyPointsRight, descriptorsRight = sift.detectAndCompute(imgRight,None)
    # Create FLANN matcher object
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams,searchParams)
    matches = flann.knnMatch(descriptorsLeft,descriptorsRight,k=2)
    
    # Apply ratio test
    goodMatches = []
    ptsLeft = []
    ptsRight = []
    
    for m, n in matches:
        
        if m.distance < 0.8 * n.distance:
            
            goodMatches.append([m])
            ptsLeft.append(keyPointsLeft[m.trainIdx].pt)
            ptsRight.append(keyPointsRight[n.trainIdx].pt)

    ptsLeft = np.int32(ptsLeft)
    ptsRight = np.int32(ptsRight)
    F, mask = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)

    print("Fundamental Matrix for combination: {}".format(camera_combination))
    print(F)
    print("")



