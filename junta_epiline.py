from __future__ import print_function

keypoints = [
  {
    "x": 545.1566772460938,
    "y": 323.4208068847656
  },
  {
    "x": 515.5092163085938,
    "y": 168.84848022460938
  },
  {
    "x": 897.8673706054688,
    "y": 234.2772979736328
  }
]

# a = -0.857495252
# b = 0.514491879
# c = 696.177652

a = 0.857495252
b = -0.514491879
c = -696.177652

for keypoint in keypoints:
  print("Quanto deveria dar: y={}".format(keypoint["y"]))
  y = (-a*keypoint["x"] - c) / b
  print("Quanto deu: y={}".format(y))
  print("") 