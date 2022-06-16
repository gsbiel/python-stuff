from __future__ import print_function

keypoints = [
  {
    "x":671.25,
    "y":408.25
  },
  {
    "x":455.5,
    "y":289.25  
  }
]

# a = -0.857495252
# b = 0.514491879
# c = 696.177652

a = -0.888722563
b = 0.458445423
c = 651.579197

for keypoint in keypoints:
  print("Quanto deveria dar: y={}".format(keypoint["y"]))
  y = (-a*keypoint["x"] - c) / b
  print("Quanto deu: y={}".format(y))
  print("") 