#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run \
    -v $SCRIPT_DIR/..:/workspace/softgym \
    -v $HOME/miniconda3:$HOME/miniconda3 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --gpus all \
    -e PATH=$HOME/miniconda3/bin:$PATH \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash
