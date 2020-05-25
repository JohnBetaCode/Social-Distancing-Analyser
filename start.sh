xhost +
docker build -t working_mage .
docker run -it -v `pwd`/configs:/usr/src/app/configs -v `pwd`/media:/usr/src/app/media -v `pwd`/python_utils:/usr/src/app/python_utils --rm --gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1  -v /tmp/.X11-unix:/tmp/.X11-unix working_mage bash