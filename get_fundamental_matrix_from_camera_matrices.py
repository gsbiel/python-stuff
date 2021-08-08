import numpy as np
import cv2 
import json

print(cv2.__version__)

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


frames = {}
camera_combinations = [("0","1"),("0","2"),("0","3"),("1","2"),("1","3"),("2","3")]
cameras = ["cam0", "cam1" , "cam2" , "cam3"]
calibration_files = ["C0_calibration.json", "C1_calibration.json", "C2_calibration.json", "C3_calibration.json"]

# Read first frame of each video
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

projection_matrix = np.array( [ [ 1, 0, 0, 0],
                                [ 0, 1, 0, 0],
                                [ 0, 0, 1, 0]] )

get_system_calibration_data()
output_file = "fundamental_matrices.npy"
fundamental_matrices = []

for camera_combination in camera_combinations:

    camLeft = camera_combination[0]
    camRight = camera_combination[1]

    intrinsic_matrices["cam{}".format(camLeft)]

    camera_left_matrix = intrinsic_matrices["cam{}".format(camLeft)].dot(projection_matrix.dot(extrinsic_matrices["cam{}".format(camLeft)]))
    camera_right_matrix = intrinsic_matrices["cam{}".format(camRight)].dot(projection_matrix.dot(extrinsic_matrices["cam{}".format(camRight)]))

    imgLeft = frames[camLeft]
    imgRight = frames[camRight]

    F = cv2.sfm.fundamentalFromProjections(camera_left_matrix,camera_right_matrix)
    fundamental_matrices.append(F)

output_dict = {}
for f_matrix, combination in zip(fundamental_matrices, camera_combinations):
    output_dict["{o}-{d}".format(o=combination[0],d=combination[1])] = f_matrix.tolist()

with open("fundamental_matrices.json", 'w') as outfile:
    json.dump(output_dict,outfile)




