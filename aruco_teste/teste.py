from __future__ import print_function
import numpy as np
from numpy.linalg import inv
import json
from cv2 import aruco
import cv2
import os
import math

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
  for i in range(0, len(cameras)):
    with open(calibration_files[i]) as file:
      data = json.load(file)
      extrinsic_matrices[cameras[i]] = np.array(data["extrinsic"]["tf"]["doubles"]).reshape((4,4))
      intrinsic_matrices[cameras[i]] = np.array(data["intrinsic"]["doubles"]).reshape((3,3))
      distortions[cameras[i]] = np.array(data["distortion"]["doubles"])
  return 

def read_json(path_to_file):
  with open(path_to_file) as f:
    return json.load(f)

def drawlines(left_frame, right_frame, lines_right, left_points, right_points):
  
  r, c = left_frame.shape

  print("")
  print("Image shape:")
  print("width: {}".format(c))
  print("height: {}".format(r))

  left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
  right_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)

  for line, left_point, right_point in zip(lines_right, left_points, right_points): 

    color = tuple(np.random.randint(0, 255, 3).tolist())
    
    # Reta
    x0, y0 = [0, -line[2]/line[1] ]
    x1, y1 = [c, (-line[0]*c - line[2])/ line[1]]

    left_point_tuple = (int(left_point[0][0]),int(left_point[0][1]))
    right_point_tuple = (int(right_point[0][0]),int(right_point[0][1]))

    right_frame = cv2.line(right_frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
    right_frame = cv2.circle(right_frame, right_point_tuple, 5, color, -1)

    left_frame = cv2.circle(left_frame, left_point_tuple, 5, color, -1)
  
  return right_frame, left_frame

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
  skew_mat = skew(T10)
  E10 = skew_mat.dot(R10)
  F10 = (inv(K1).transpose()).dot(E10.dot(inv(K0)))
  return F10

def compute_correspond_epilines(points, F):
  lines = []
  for point in points:
    p = np.array([[point[0][0], point[0][1], 1]])
    line = F.dot(p.T)
    line = line.T
    k = math.sqrt((math.pow(line[0][0],2) + math.pow(line[0][1],2)))
    line = line/k 
    lines.append(line)
  return lines

def shortest_distance(x1, y1, a, b, c):  
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d

def get_camera_params(cam_identifier, distortions, intrinsic_matrices, extrinsic_matrices):
  dist = distortions["cam{}".format(cam_identifier)]
  K = intrinsic_matrices["cam{}".format(cam_identifier)]
  mRT =  extrinsic_matrices["cam{}".format(cam_identifier)] 
  return dist, K, mRT

def detect_aruco_in_frame(frame):
  parameters =  aruco.DetectorParameters_create()
  aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
  corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
  if len(corners) < 1:
    return -1
  return corners[0][0]

def calculate_aruco_midpoint(corners):
  quinas = corners.flatten().reshape(-1,2).T
  midpoint = [[np.mean(quinas[0]),np.mean(quinas[1])]]
  return midpoint

######################################################################################################################################
# MAIN CODE

def main():

  # define qual é a câmera da esquerda e qual é a câmera da direita
  left_cam_ref = 0
  right_cam_ref = 1

  # Lê os dados de calibração e armazena eles nas estruturas de dados definidas lá no começo do código
  get_system_calibration_data()

  # Lê os parâmetros das câmeras da esquerda e da direita
  [dist0, K0, mRT0] = get_camera_params(left_cam_ref, distortions, intrinsic_matrices, extrinsic_matrices)
  [dist1, K1, mRT1] = get_camera_params(right_cam_ref, distortions, intrinsic_matrices, extrinsic_matrices)

  # Lê o frame da câmera da esquerda e da direita (o primeiro frame de cada uma)
  frame_left = first_frame_debug["{}".format(left_cam_ref)]
  frame_right = first_frame_debug["{}".format(right_cam_ref)]

  # Calcula a matriz fundamental
  F = fundamental_matrix(K0, mRT0, K1, mRT1)

  print("Fundamental Matrix")
  print(f"F = {F}")
  print("")

  # Detecta o aruco nos frames da esquerda e da direita
  aruco_corners_in_left_frame = detect_aruco_in_frame(frame_left)
  aruco_corners_in_right_frame = detect_aruco_in_frame(frame_right)

  # Calcula o ponto medio do aruco nos frames da esquerda e da direita
  # Também reformata o array para ficar de acordo com o que a função cv2.computeCorrespondEpilines espera
  aruco_midpoint_in_left_frame = calculate_aruco_midpoint(aruco_corners_in_left_frame)
  aruco_midpoint_in_left_frame = np.array(aruco_midpoint_in_left_frame).reshape(-1,1,2)

  aruco_midpoint_in_right_frame = calculate_aruco_midpoint(aruco_corners_in_right_frame)
  aruco_midpoint_in_right_frame = np.array(aruco_midpoint_in_right_frame).reshape(-1,1,2)

  print(f"Aruco - Left point: {aruco_midpoint_in_left_frame}")
  print(f"Aruco - Right epiline: {aruco_midpoint_in_right_frame}")

  # Transforma o ponto médio do aruco na imagem da esquerda em uma linha epipolar a ser
  # plotada na imagem da direita
  linesRight = cv2.computeCorrespondEpilines(aruco_midpoint_in_left_frame, 1, F)
  linesRight = linesRight.reshape(-1,3)

  # Calcula a distância entre o ponto médio do aruco na imagem da direita e a linha epipolar calculada acima
  right_point_distance = shortest_distance(aruco_midpoint_in_right_frame[0][0][0], aruco_midpoint_in_right_frame[0][0][1], linesRight[0][0], linesRight[0][1], linesRight[0][2])
  print(f"Distance between corresponding point and epipolar line: {right_point_distance}")

  # Desenha nos frames os pontos do aruco e a linha epipolar
  imagem_a, imagem_b = drawlines(frame_left, frame_right, linesRight, aruco_midpoint_in_left_frame, aruco_midpoint_in_right_frame)

  # Exibe a imagem
  Hori = np.concatenate((imagem_b, imagem_a), axis=1)
  imS = cv2.resize(Hori, (1300, 640))
  cv2.imshow('HORIZONTAL', imS)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

######################################################################################################################################
# ENTRYPOINT
main()