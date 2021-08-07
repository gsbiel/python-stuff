import os
import numpy as np
import json
import cv2

from time import sleep

#####################################################################################################################################
# DEBUGGING PURPOSE

first_frame_debug = {}

cap = cv2.VideoCapture("./p001g15/p001g15c00.mp4")
ret, frame = cap.read()
first_frame_debug["0"] = frame
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c01.mp4")
ret, frame = cap.read()
first_frame_debug["1"] = frame
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c02.mp4")
ret, frame = cap.read()
first_frame_debug["2"] = frame
cap.release()

cap = cv2.VideoCapture("./p001g15/p001g15c03.mp4")
ret, frame = cap.read()
first_frame_debug["3"] = frame
cap.release()

######################################################################################################################################
# GLOBAL VARIABLES

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

######################################################################################################################################
# MAIN CODE

def main():
    path = "./detections"
    list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
    for subfolder in list_subfolders_with_paths:
        print("Processing detections in folder: {path}/output/".format(path=subfolder))
        list_files_path = [f.path for f in os.scandir(subfolder+"/output") if f.is_file()]
        path_to_each_camera_detection_file = identify_path_for_each_camera_detection_file(list_files_path)
        camera_detection_data = {}
        camera_detection_data["0"] = read_json(path_to_each_camera_detection_file["0"])
        camera_detection_data["1"] = read_json(path_to_each_camera_detection_file["1"])
        camera_detection_data["2"] = read_json(path_to_each_camera_detection_file["2"])
        camera_detection_data["3"] = read_json(path_to_each_camera_detection_file["3"])

        for cam0_annotation, cam1_annotation, cam2_annotation, cam3_annotation in  zip(
                                                                                        camera_detection_data["0"]["annotations"],
                                                                                        camera_detection_data["1"]["annotations"],
                                                                                        camera_detection_data["2"]["annotations"],
                                                                                        camera_detection_data["3"]["annotations"]):

            print(cam0_annotation["frame_id"])
            print(cam1_annotation["frame_id"])
            print(cam2_annotation["frame_id"])
            print(cam3_annotation["frame_id"])
            print("")

            # To run the code in a single frame
            break

        # To run the code in a single sub folder
        break

    return

######################################################################################################################################
# ENTRYPOINT
main()