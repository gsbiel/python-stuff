import os
import numpy as np
import json
import cv2
from matplotlib import pyplot as plt
from time import sleep

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

######################################################################################################################################
# GLOBAL VARIABLES

cameras = ["cam0", "cam1" , "cam2" , "cam3"]

calibration_files = ["C0_calibration.json", "C1_calibration.json", "C2_calibration.json", "C3_calibration.json"]

camera_combinations = [("0","1"),("0","2"),("0","3"),("1","2"),("1","3"),("2","3")]

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

######################################################################################################################################
# AUX FUNCTIONS

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

def identify_path_for_each_camera_detection_file(list_files_path):
    cam_patterns = ["c00", "c01", "c02", "c03"]
    camera_detection_files = {}
    for i in range(0, len(cam_patterns)):
        for file_path in list_files_path:
            if cam_patterns[i] in file_path:
                camera_detection_files["{}".format(i)] = file_path
    return camera_detection_files

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
        
        x0, y0 = map(int, [0, -line[0][2] / line[0][1] ])
        x1, y1 = map(int, [c, -(line[0][2] + line[0][0] * c) / line[0][1] ])
        
        point_tuple = (int(pt2[0][0]),int(pt2[0][1]))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, point_tuple, 5, color, -1)
    return img1, img2

######################################################################################################################################
# MAIN CODE

def main():
    path = "./detections"
    fundamental_matrices = read_json("fundamental_matrices.json")
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

            # print(cam0_annotation["frame_id"])
            # print(cam1_annotation["frame_id"])
            # print(cam2_annotation["frame_id"])
            # print(cam3_annotation["frame_id"])

            # ------------------------------------------------------------------------------------------------------------------------
            
            F = np.array(fundamental_matrices["0-1"]).reshape(3,3)
            print(F)
            for skeleton in cam0_annotation["objects"]:
                left_points = []
                for keypoint in skeleton["keypoints"]:
                    left_points.append([keypoint["position"]["x"],keypoint["position"]["y"]])
                
                left_points = np.array(left_points).reshape(-1,1,2)
                linesRight = cv2.computeCorrespondEpilines(left_points, 1, F)

                imagem_a, imagem_b = drawlines(first_frame_debug["1"], first_frame_debug["0"], linesRight, left_points)

                Hori = np.concatenate((imagem_b, imagem_a), axis=1)
                Verti = np.concatenate((imagem_b, imagem_a), axis=0)
                cv2.imshow('HORIZONTAL', Hori)
                # cv2.imshow('HORIZONTAL', Verti)
                # cv2.imshow("imagem", imagem_a)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # -------------------------------------------------------------------------------------------------------------------------
            # To run the code in a single frame
            break

        # To run the code in a single sub folder
        break

    return

######################################################################################################################################
# ENTRYPOINT
main()

# REF 
# https://www.geeksforgeeks.org/python-opencv-epipolar-geometry/
# https://stackoverflow.com/questions/51089781/how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv
# http://amroamroamro.github.io/mexopencv/matlab/cv.computeCorrespondEpilines.html
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1