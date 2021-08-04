import os
f = open("process.sh", "a")

for person in ["p001", "p002"]:
  for gesture in ["g01","g02","g03","g04","g05","g06","g07","g08","g09","g10","g11","g12","g13","g14","g15"]:
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" scp /home/gaspar/Documents/UFES/openpose/videos-para-avaliar/{person}{gesture}/{person}{gesture}c* gaspar@zeus:/home/gaspar/Documents/openpose/videos/input'.format(person=person, gesture=gesture))
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" ssh gaspar@zeus "docker container run --rm --gpus all --name gasper_openpose -v /home/gaspar/Documents/openpose/videos/input:/openpose/examples/media/videos/input -v /home/gaspar/Documents/openpose/videos/output:/openpose/examples/media/videos/output -v /home/gaspar/Documents/repos/openpose/examples/tutorial_api_python:/openpose/examples/tutorial_api_python -e DISPLAY gsbiel/openpose_gpu:0.4 python3 /openpose/examples/tutorial_api_python/body_from_video.py"')
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" scp -r gaspar@zeus:/home/gaspar/Documents/openpose/videos/output/ /home/gaspar/Documents/UFES/openpose/videos-para-avaliar/{person}{gesture}/'.format(person=person, gesture=gesture))
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" ssh gaspar@zeus rm -rfv ~/Documents/openpose/videos/input/')
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" ssh gaspar@zeus mkdir ~/Documents/openpose/videos/input')
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" ssh gaspar@zeus "docker container run --rm --gpus all --name gasper_openpose -v /home/gaspar/Documents/openpose/videos/input:/openpose/examples/media/videos/input -v /home/gaspar/Documents/openpose/videos/output:/openpose/examples/media/videos/output -v /home/gaspar/Documents/repos/openpose/examples/tutorial_api_python:/openpose/examples/tutorial_api_python -e DISPLAY gsbiel/openpose_gpu:0.4 rm -r /openpose/examples/media/videos/output/"')
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" ssh gaspar@zeus rm -rfv ~/Documents/openpose/videos/output/')
    f.write("\n")
    f.write('sshpass -f "/home/gaspar/Documents/UFES/openpose/password" ssh gaspar@zeus mkdir ~/Documents/openpose/videos/output')
    f.write("\n")

