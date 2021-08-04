import os
import json
import cv2
import numpy as np

#########################################################################################
# GLOBAL VARIABLES

# Total amount of keypoints presented in the new OpenPose model
KEYPOINTS_TOTAL = 25.0

# A keypoint is considered as a valid one if its score is greater than this value
SCORE_TRIGGER = 0.6

# Percentage of valid keypoints an object must have to be considered as a human skeleton
VALID_KEYPOINTS_TRIGGER = 0.4

# Openpose Mapping values

KEYPOINT_COLORS = {
    "0":(195,1,68),
    "1":(206,37,9),
    "2":(186,60,1),
    "3":(169,115,1),
    "4":(151,153,1),
    "5":(143,213,4),
    "6":(143,213,4),
    "7":(143,213,4),
    "8":(206,37,9),
    "9":(143,213,4),
    "10":(3,196,134),
    "11":(0,228,227),
    "12":(0,98,154),
    "13":(1,51,158),
    "14":(1,51,158),
    "15":(225,1,155),
    "16":(101,0,152),
    "17":(148,1,154),
    "18":(60,0,199),
    "19":(1,51,158),
    "20":(1,51,158),
    "21":(1,51,158),
    "22":(0,228,227),
    "23":(0,228,227),
    "24":(0,228,227),
}

KEYPOINTS_MAPPING = {
    "0":[{"id": 15, "color": (138,1,91)},{"id": 16, "color": (100,1,150)},{"id": 1, "color": (153,1,52)}],
    "1":[{"id": 0, "color": (153,1,52)},{"id": 2, "color": (154,50,1)},{"id": 5, "color": (94,145,0)},{"id": 8, "color": (152,0,0)}],
    "2":[{"id": 1, "color": ((152,50,0))},{"id": 3, "color": (154,102,0)}],
    "3":[{"id": 2, "color": (154,102,0)},{"id": 4, "color": (153,155,1)}],
    "4":[{"id": 3, "color": (153,155,1)}],
    "5":[{"id": 1, "color": (94,145,0)},{"id": 6, "color": (51,152,1)}],
    "6":[{"id": 5, "color": (51,152,1)},{"id": 7, "color": (0,153,0)}],
    "7":[{"id": 6, "color": (0,153,0)}],
    "8":[{"id": 1, "color": (152,0,0)},{"id": 9, "color": (1,153,52)},{"id": 12, "color": (0,101,153)}],
    "9":[{"id": 8, "color": (1,153,52)},{"id": 10, "color": (0,152,101)}],
    "10":[{"id": 9, "color": (0,152,101)},{"id": 11, "color": (0,153,153)}],
    "11":[{"id": 10, "color": (0,153,153)},{"id": 22, "color": (8,149,153)},{"id": 24, "color": (8,149,153)}],
    "12":[{"id": 8, "color": (0,101,153)},{"id": 13, "color": (0,49,144)}],
    "13":[{"id": 12, "color": (0,49,144)},{"id": 14, "color": (0,0,152)}],
    "14":[{"id": 13, "color": (0,0,152)},{"id": 19, "color": (0,0,152)},{"id": 21, "color": (0,0,152)}],
    "15":[{"id": 0, "color": (138,1,91)},{"id": 17, "color": (155,1,155)}],
    "16":[{"id": 0, "color": (100,1,150)},{"id": 18, "color": (50,1,152)}],
    "17":[{"id": 15, "color": (155,1,155)}],
    "18":[{"id": 16, "color": (50,1,152)}],
    "19":[{"id": 14, "color": (0,0,152)},{"id": 20, "color": (1,0,140)}],
    "20":[{"id": 19, "color": (1,0,140)}],
    "21":[{"id": 14, "color": (0,0,152)}],
    "22":[{"id": 23, "color": (8,149,153)},{"id": 11, "color": (8,149,153)}],
    "23":[{"id": 22, "color": (8,149,153)}],
    "24":[{"id": 11, "color": (8,149,153)}]
}

#########################################################################################
# AUX FUNCTIONS

def read_json(json_path):
    with open(json_path) as f:
        return json.load(f)

def read_frames(video_path):
    input_frames = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        # Read frame
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            # Store frame in array
            input_frames.append(frame)
        else:
            break
    cap.release()
    return input_frames

def write_video(video_path, frames):
    out = cv2.VideoWriter(video_path, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1288,728))
    for frame in frames:
        out.write(frame)
    out.release()

def draw_keypoints(input_frames, json_object):
    plotted_frames = []
    for frame, frame_annotation in zip(input_frames, json_object["annotations"]):
        plotted_image = np.copy(frame)
        for frame_object in frame_annotation["objects"]:
            plotted_ids = []
            keypoints_dict = {}
            for keypoint in frame_object["keypoints"]:
                x = keypoint["position"]["x"]
                y = keypoint["position"]["y"]
                keypoint_id = keypoint["id"]
                keypoints_dict[keypoint_id]={
                    "x":int(x),
                    "y":int(y)
                }
                plotted_ids.append(keypoint_id)
                plotted_image = cv2.circle(plotted_image, (int(x),int(y)), radius=8, color=KEYPOINT_COLORS[keypoint_id], thickness=-1)
            plotted_paths = []
            for plotted_id in plotted_ids:
                for mapping in KEYPOINTS_MAPPING[plotted_id]:
                    if str(mapping["id"]) in plotted_ids:
                        if "{plotted}-{mapped}".format(plotted=plotted_id,mapped=mapping["id"]) not in plotted_paths and "{mapped}-{plotted}".format(plotted=plotted_id,mapped=mapping["id"]) not in plotted_paths:
                            x1 =  keypoints_dict[plotted_id]["x"]
                            y1 =  keypoints_dict[plotted_id]["y"]
                            x2 =  keypoints_dict[str(mapping["id"])]["x"]
                            y2 =  keypoints_dict[str(mapping["id"])]["y"]
                            plotted_image = cv2.line(plotted_image, (x1, y1), (x2, y2), mapping["color"], thickness=2)
                            plotted_paths.append("{plotted}-{mapped}".format(plotted=plotted_id,mapped=mapping["id"]))
        plotted_frames.append(plotted_image)
    return plotted_frames

def draw_keypoints_on_video(video_path, json_object):

    print("")
    print("Reading frames...")
    input_frames = read_frames(video_path)
    print("Done! {} frames have been read.".format(len(input_frames)))

    print("")
    print("Drawing keypoints on input frames...")
    plotted_frames = draw_keypoints(input_frames, json_object)

    print("")
    print("Writing output video file...")
    write_video("output_video.avi", plotted_frames)


#########################################################################################
# MAIN

def main():

    json_filename = "p002g15c03"
    json_path = "./{}.json".format(json_filename)
    video_path = "./{}.mp4".format(json_filename)

    json_obj = read_json(json_path)
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

    # Generate video with filtered keypoints plotted on it.
    draw_keypoints_on_video(video_path, json_obj)

    # Serializing json 
    json_serialized = json.dumps(json_obj)
    
    # Writing to sample.json
    with open("{}-filtered.json".format(json_filename), "w") as outfile:
        outfile.write(json_serialized)

    return

#########################################################################################
# ENTRYPOINT
main()