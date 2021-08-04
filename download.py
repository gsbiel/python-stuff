import os
f = open("download.sh", "a")

for person in ["p001", "p002"]:
  for gesture in ["g01","g02","g03","g04","g05","g06","g07","g08","g09","g10","g11","g12","g13","g14","g15"]:
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" scp gaspar@zeus:/public/datasets/ufes-2020-01-23/{person}{gesture}_output.mp4 /home/gaspar/Documents/UFES/openpose/videos-para-avaliar/{person}{gesture} '.format(person=person, gesture=gesture))
    f.write("\n")

    dirName = "/home/gaspar/Documents/UFES/openpose/videos-para-avaliar/{person}{gesture}".format(person=person, gesture=gesture)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

    for camera in ["c00","c01","c02","c03"]:
      f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" scp gaspar@zeus:/public/datasets/ufes-2020-01-23/{person}{gesture}{camera}.mp4 /home/gaspar/Documents/UFES/openpose/videos-para-avaliar/{person}{gesture} '.format(person=person, gesture=gesture, camera=camera))
      f.write("\n")

