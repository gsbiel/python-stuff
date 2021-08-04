# coding: utf-8
import sys
import cv2
import os
from sys import platform
import argparse
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
    print(frame)
    exit()
    objects_json = []
    resolution_json = {
      "height": np.array(frame).shape[1],
      "width": np.array(frame).shape[0]
    }
    for skeleton in frame:
      keypoints_json = []
      keypoint_id = 0
      for keypoint in skeleton:
        if keypoint != [0.0, 0.0, 0.0]:
          keypoint_json = {
            "id": "{id}".format(id = keypoint_id),
            "score": keypoint[2],
            "position": {
              "x": keypoint[0],
              "y": keypoint[1],
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

with open('/home/gaspar/Documents/UFES/openpose/videos/output/output/p001g15c00.json') as json_file:
    data_json = json.load(json_file)
    return_data = get_json_data(data_json[data_json.keys()[0]])
    with open( './output.json', 'w') as outfile:
      json.dump(return_data, outfile)
    



