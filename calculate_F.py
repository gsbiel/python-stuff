from __future__ import print_function
import numpy as np
from numpy.linalg import inv
import json
import cv2
import os

#####################################################################################################################################
# DEBUGGING PURPOSE

first_frame_debug = {}

cap = cv2.VideoCapture("./p001g15/p001g15c00.mp4")
ret, frame = cap.read()
first_frame_debug["0"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c01.mp4")
ret, frame = cap.read()
first_frame_debug["1"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c02.mp4")
ret, frame = cap.read()
first_frame_debug["2"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c03.mp4")
ret, frame = cap.read()
first_frame_debug["3"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

############################################################################################################

cameras = ["cam0", "cam1" , "cam2" , "cam3"]
calibration_files = ["C0_calibration.json", "C1_calibration.json", "C2_calibration.json", "C3_calibration.json"]

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

# Total amount of keypoints presented in the new OpenPose model
KEYPOINTS_TOTAL = 25.0

# A keypoint is considered as a valid one if its score is greater than this value
SCORE_TRIGGER = 0.6

# Percentage of valid keypoints an object must have to be considered as a human skeleton
VALID_KEYPOINTS_TRIGGER = 0.4

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

def filter_keypoints(json_obj):
  for annotation in json_obj["annotations"]: 
    frame_resolution = annotation["resolution"]
    frame_id = annotation["frame_id"]
    filtered_objects = []
    for object_item in annotation["objects"]:
      valid_keypoints = []
      for keypoint in object_item["keypoints"]:
        keypoint_id = keypoint["id"]
        keypoint_score = keypoint["score"]
        keypoint_position = keypoint["position"]
        if keypoint_score > SCORE_TRIGGER:
          valid_keypoints.append(keypoint)
      if float(len(valid_keypoints))/KEYPOINTS_TOTAL > VALID_KEYPOINTS_TRIGGER:
        filtered_objects.append({
          "label":"0",
          "id":"0",
          "score":"0.0",
          "keypoints":valid_keypoints
        })
    annotation["objects"] = filtered_objects
  return json_obj

def identify_path_for_each_camera_detection_file(list_files_path):
  cam_patterns = ["c00", "c01", "c02", "c03"]
  camera_detection_files = {}
  for i in range(0, len(cam_patterns)):
    for file_path in list_files_path:
      if cam_patterns[i] in file_path:
        camera_detection_files["{}".format(i)] = file_path
  return camera_detection_files 

def read_json(path_to_file):
  with open(path_to_file) as f:
    return json.load(f)

def drawlines_v2(img1, img2, lines, pts2):
  r, c = img1.shape
  print("Image shape:")
  print("width: {}".format(r))
  print("height: {}".format(c))
  img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
  img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
  for line, pt2 in zip(lines, pts2):   
    color = tuple(np.random.randint(0, 255,
                                    3).tolist())
    x0, y0 = map(int, [0, -line[0][2] / line[0][1] ])
    x1, y1 = map(int, [c, -(line[0][2] + line[0][0] * c) / line[0][1] ])
    point_tuple = (int(pt2[0][0]),int(pt2[0][1]))
    img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
    img2 = cv2.circle(img2, point_tuple, 5, color, -1)
  return img1, img2

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

  path = "./detections"
  get_system_calibration_data()
  K0 = intrinsic_matrices["cam{}".format(left_cam_ref)]
  K1 = intrinsic_matrices["cam{}".format(right_cam_ref)]
  mRT0 =  extrinsic_matrices["cam{}".format(left_cam_ref)]
  mRT1 = extrinsic_matrices["cam{}".format(right_cam_ref)]
  F = fundamental_matrix(K0, mRT0, K1, mRT1)
  print("Fundamental Matrix: {}".format(F))
  list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
  for subfolder in list_subfolders_with_paths:
    print("Processing detections in folder: {path}/output/".format(path=subfolder))
    list_files_path = [f.path for f in os.scandir(subfolder+"/output") if f.is_file()]
    path_to_each_camera_detection_file = identify_path_for_each_camera_detection_file(list_files_path)
    camera_detection_data = {}
    camera_detection_data["0"] = filter_keypoints(read_json(path_to_each_camera_detection_file["0"]))
    camera_detection_data["1"] = filter_keypoints(read_json(path_to_each_camera_detection_file["1"]))
    camera_detection_data["2"] = filter_keypoints(read_json(path_to_each_camera_detection_file["2"]))
    camera_detection_data["3"] = filter_keypoints(read_json(path_to_each_camera_detection_file["3"]))
    for cam0_annotation, cam1_annotation, cam2_annotation, cam3_annotation in  zip(
                                                                                    camera_detection_data["0"]["annotations"],
                                                                                    camera_detection_data["1"]["annotations"],
                                                                                    camera_detection_data["2"]["annotations"],
                                                                                    camera_detection_data["3"]["annotations"]):
      for skeleton in cam0_annotation["objects"]:
        left_points = []
        # for keypoint in skeleton["keypoints"]:
        #     left_points.append([keypoint["position"]["x"],keypoint["position"]["y"]])
        #     break
        left_points.append([skeleton["keypoints"][0]["position"]["x"],skeleton["keypoints"][0]["position"]["y"]])
        print("ID da junta: {}".format(skeleton["keypoints"][0]["id"]))
        
        # left_points.append([skeleton["keypoints"][1]["position"]["x"],skeleton["keypoints"][1]["position"]["y"]])
        # left_points.append([skeleton["keypoints"][2]["position"]["x"],skeleton["keypoints"][2]["position"]["y"]])
        # left_points.append([skeleton["keypoints"][3]["position"]["x"],skeleton["keypoints"][3]["position"]["y"]])
        # left_points.append([skeleton["keypoints"][4]["position"]["x"],skeleton["keypoints"][4]["position"]["y"]])

        left_points = np.array(left_points).reshape(-1,1,2)
        linesRight = cv2.computeCorrespondEpilines(left_points, 1, F)

        print(f"Left point: {left_points}")
        print(f"Right epiline: {linesRight}")

        imagem_a, imagem_b = drawlines(first_frame_debug["{}".format(right_cam_ref)], first_frame_debug["{}".format(left_cam_ref)], linesRight, left_points)
        Hori = np.concatenate((imagem_b, imagem_a), axis=1)
        imS = cv2.resize(Hori, (1300, 640))
        cv2.imshow('HORIZONTAL', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
      break
    break
  return

######################################################################################################################################
# ENTRYPOINT

main()




