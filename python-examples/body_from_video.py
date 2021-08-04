# coding: utf-8
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pyopenpose as op
import numpy as np
import json
from datetime import datetime

def get_current_datetime():
  current_date = datetime.today()
  date_string = "{year}-{month}-{day}T{hour}:{minute}:{second}".format(
    year=current_date.year,
    month=current_date.month,
    day=current_date.day,
    hour=current_date.hour,
    minute=current_date.minute,
    second=current_date.microsecond/1000.0
  )
  return date_string

# frame_list são os keypoints de todos os esqueletos identificados no frame da imagem
# que está sendo processada.
def get_json_data(frame_list, image_width, image_height):
  
  output_json = {
    "annotations":[],
    "created_at": get_current_datetime()
  }

  frameid_json = 0
  for frame in frame_list:
    objects_json = []
    resolution_json = {
      "height": image_height,
      "width": image_width
    }
    for skeleton in frame:
      keypoints_json = []
      keypoint_id = 0
      for keypoint in skeleton:
        if keypoint.tolist() != [0.0, 0.0, 0.0]:
          keypoint_json = {
            "id": "{id}".format(id = keypoint_id),
            "score": float(keypoint[2]),
            "position": {
              "x": float(keypoint[0]),
              "y": float(keypoint[1]),
              "z": 0.0
            }
          }
          keypoints_json.append(keypoint_json)
        keypoint_id += 1

      objects_json.append({
        "keypoints": keypoints_json,
        "label": "",
        "id": "0",
        "score": 0.0
      })
    output_json["annotations"].append({
      "objects": objects_json,
      "resolution": resolution_json,
      "frame_id": frameid_json
    })
    frameid_json += 1
  
  return output_json


###########################################################################################################
# MAIN CODE

try:

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="/openpose/examples/media/videos/input", help="Video to be processed is stored here.")
    parser.add_argument("--output_folder", default="/openpose/examples/media/videos/output", help="Processed video is stored here.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/openpose/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Get path to video files and prepare data structure
    input_frames_dict = {}
    output_frames_dict = {}
    output_keypoints_dict = {}
    videos_path = []
    for path in os.listdir(args[0].input_folder):
      full_path = os.path.join(args[0].input_folder, path)
      if os.path.isfile(full_path):
          if ".mp4" in path:
            videos_path.append(full_path)
            input_frames_dict[full_path] = []
            output_frames_dict[full_path] = []
            output_keypoints_dict[full_path] = []

    print("")        
    for video_path in videos_path:
      print("Reading frames for: {}".format(video_path))
      cap = cv2.VideoCapture(video_path)
      while(cap.isOpened()):
        # Read frame
        ret, frame = cap.read()

        if isinstance(frame, np.ndarray):
          # Store frame in array
          input_frames_dict[video_path].append(frame)
        else:
          break
      cap.release()
      print("Done! {} frames have been read.".format(len(input_frames_dict[video_path])))

    print("")
    for video_path in input_frames_dict:
      print("Processing frames for: {}".format(video_path))
      total_frames = len(input_frames_dict[video_path])
      for frame in input_frames_dict[video_path]:
        # Process Frame
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        output_frames_dict[video_path].append(datum.cvOutputData)
        output_keypoints_dict[video_path].append(datum.poseKeypoints)
      print("Done!")

    print("")
    for video_path in output_frames_dict:
      output_path = "{path}/{file}".format(path=args[0].output_folder, file="{}.avi".format(video_path.split("/")[-1].split(".")[0])) 
      print("Writing frames to video file: {}".format(output_path))
      # Get the size of the frame
      height, width, layers = output_frames_dict[video_path][0].shape
      size = (width,height)
      fps = 25.0
      out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
      for frame in output_frames_dict[video_path]:
        out.write(frame)
      out.release()
    print("Done!")

    print("")
    for video_path in output_keypoints_dict:
      output_path = "{path}/{file}".format(path=args[0].output_folder, file="{}.json".format(video_path.split("/")[-1].split(".")[0])) 
      print("Writing JSON file: {}".format(output_path))
      image_shape = input_frames_dict[video_path][0].shape
      dict_data = get_json_data(output_keypoints_dict[video_path], image_shape[1], image_shape[0])
      with open( output_path, 'w') as outfile:
        json.dump(dict_data, outfile)
    print("Done!")
    
except Exception as e:
    print(e)
    sys.exit(-1)
