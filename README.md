# From Flab to Ab: A Smart Home Gym

This project contains the source code for our Computer Engineering Senior
Project, satisfying the requirements for ECE 4710 at the University of Utah.
The project includes backend code for pose estimation, frontend code for display
on the mirror, and a Flask server for communication with the iOS application
that accompanies this project.

## Overview

The purpose of this project is to provide real-time visual feedback and
auto-logging capability to a user who is exercising. The project currently
supports three exercises: bicep curl, shoulder press, and squat.

<img
src="media/curl.gif"
height=256/>

<img
src="media/press.gif"
height=256/>

<img
src="media/squat.gif"
height=256/>

## Dependencies

This project makes use of the [TensorRT Pose
Estimation](https://github.com/zhangzhe1103/trt_pose) library for model
inference accelerated by NVIDIA TensorRT. Please refer to relevant documentation
their for full installation instructions.

### Models

We used the following followed which was pre-trained on the MSCOCO dataset.

| Model | Jetson Nano | Weights |
|-------|-------------|---------|
| resnet18_baseline_att_224x224_A | 14 | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |


## Running

To run this application, navigate to the [tasks/human_pose](tasks/human_pose)
directory and run the [get_video.py](tasks/human_pose/get_video.py) script.

- `python3 get_video.py`

In addition to loading the model, running this script also starts up a server
running on port 5000. To begin an exercise, connect to this server and hit one
of the supported HTTP endpoints using a browser or the iOS application that
accompanies this project.

### Endpoints
  - _startSession_: Begin a workout session
  - _endSession_: End a workout session
  - _rightCurl_: Initiate right bicep curl exercise
  - _leftCurl_: Initiate left bicep curl exercise
  - _shoulderPress_: Initiate shoulder press exercise
  - _squat_: Initiate squat exercise
