from __future__ import print_function
import numpy as np
from numpy.linalg import inv
import json
import cv2
import os

#####################################################################################################################################
# READ THE FIRST FRAME OF EACH VIDEO

first_frame_debug = {}

cap = cv2.VideoCapture("./camera-00.mp4")
ret, frame = cap.read()
first_frame_debug["0"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./camera-01.mp4")
ret, frame = cap.read()
first_frame_debug["1"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./camera-02.mp4")
ret, frame = cap.read()
first_frame_debug["2"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./camera-03.mp4")
ret, frame = cap.read()
first_frame_debug["3"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

############################################################################################################
# INITIALIZE VARIABLES

cameras = ["cam0", "cam1" , "cam2" , "cam3"]
calibration_files = ["0.json", "1.json", "2.json", "3.json"]

intrinsic_matrices = {
  "cam0": np.zeros((3,3)),
  "cam1": np.zeros((3,3)),
  "cam2": np.zeros((3,3)),
  "cam3": np.zeros((3,3)),
}

extrinsic_matrices = {
  "cam0": np.zeros((4,4)),
  "cam1": np.zeros((4,4)),
  "cam2": np.zeros((4,4)),
  "cam3": np.zeros((4,4)),
}

distortions = {
  "cam0": np.zeros((1,5)),
  "cam1": np.zeros((1,5)),
  "cam2": np.zeros((1,5)),
  "cam3": np.zeros((1,5)),
}

def get_system_calibration_data():
  global intrinsic_matrices
  global extrinsic_matrices
  global distortions
  for i in range(len(cameras)):
    with open(calibration_files[i]) as file:
      data = json.load(file)
      extrinsic_matrices[cameras[i]] = np.array(data["extrinsic"]["tf"]["doubles"]).reshape((4,4))
      intrinsic_matrices[cameras[i]] = np.array(data["intrinsic"]["doubles"]).reshape((3,3))
      distortions[cameras[i]] = np.array(data["distortion"]["doubles"])
  return 

def read_json(path_to_file):
  with open(path_to_file) as f:
    return json.load(f)

def drawlines(img1, img2, lines, pts2):
  r, c = img1.shape
  print("Image shape:")
  print("width: {}".format(r))
  print("height: {}".format(c))
  img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
  img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
  for line, pt2 in zip(lines, pts2): 
    color = tuple(np.random.randint(0, 255,
                                    3).tolist())
    #y = (-line[0][0]*x - line[0][2]) / line[0][1]
    x0, y0 = [0, -line[0][2]/line[0][1] ]
    x1, y1 = [c, (-line[0][0]*c - line[0][2])/ line[0][1]] 
    point_tuple = (int(pt2[0][0]),int(pt2[0][1]))
    img1 = cv2.line(img1, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
    img2 = cv2.circle(img2, point_tuple, 5, color, -1)
  return img1, img2

def skew(vector):
  if len(vector) != 3:
    raise Exception("skew(): vector entry has not size of 3")
  
  return np.array([[0.0,         -vector[2], vector[1]],
                  [vector[2],   0.0,        -vector[0]],
                  [-vector[1],  vector[0],  0.0]])

def fundamental_matrix(K0, mRT0, K1, mRT1):
  mRT10 = mRT1.dot(inv(mRT0))
  R10 = mRT10[0:3,0:3] 
  T10 = mRT10[0:3,-1]
  E10 = skew(T10).dot(R10)
  F10 = (inv(K1).transpose()).dot(E10.dot(inv(K0)))
  return F10

######################################################################################################################################
# MAIN CODE

def main():
  left_cam_ref = 0
  right_cam_ref = 1
  get_system_calibration_data()
  K0 = intrinsic_matrices["cam{}".format(left_cam_ref)]
  K1 = intrinsic_matrices["cam{}".format(right_cam_ref)]
  mRT0 =  extrinsic_matrices["cam{}".format(left_cam_ref)]
  mRT1 = extrinsic_matrices["cam{}".format(right_cam_ref)]
  F = fundamental_matrix(K0, mRT0, K1, mRT1)
  print(f"F = {F}")

  # Instância objetos usados pelo detector do Aruco
  parameters =  cv2.aruco.DetectorParameters_create()
  aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_4X4_50)

######################################################################################################################################
# ENTRYPOINT

main()