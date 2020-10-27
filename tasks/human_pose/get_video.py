import argparse
import json
import os.path
import time


import threading


import cv2
import PIL.Image
import torch
import torch2trt
import torchvision.transforms as transforms
from torch2trt import TRTModule

import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects 


from src.camera import Camera
from src.helper import draw, preprocess, WIDTH, HEIGHT
from src.model import Model

from flask import Flask
from flask_restful import Api, Resource

from src.api import CurlAPI

executing = False


# def LeftBicepCurl():

#     curl = LeftBicepCurl()
   
#     executing = True



#     return 





def main():

    print("Beginning script")
    # Load the annotation file and create a topology tensor
    with open("human_pose.json", "r") as f:
        human_pose = json.load(f)

    # Create a topology tensor (intermediate DS that describes part linkages)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    
    # Construct and load the model
    model = Model(pose_annotations=human_pose)
    model.load_model("resnet")
    
    model.get_optimized()
    model.log_fps()
    print("Set up model")

    # Set up the camera
    camera = Camera(width=640, height=480)
    camera.capture_video("mp4v", "/tmp/output.mp4")
    assert camera.cap is not None, "Camera Open Error"
    print("Set up camera")

    # Set up callable class used to parse the objects from the neural network
    parse_objects = ParseObjects(topology)  # from trt_pose.parse_objects

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(CurlAPI, '/curl')
    app.run(threaded=True)

    while not executing:
        pass
    
    print("Executing...")
    # Execute while the camera is open and we haven't reached the time limit

    exit()
    count = 0
    time_limit = 200
    while camera.cap.isOpened() and count < time_limit:
        t = time.time()
        succeeded, image = camera.cap.read()
        if not succeeded:
            print("Camera read Error")
            break

        resized_img = cv2.resize(
            image, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
        )
        preprocessed = preprocess(resized_img)
        counts, objects, peaks = model.execute_neural_net(
            data=preprocessed, parser=parse_objects
        )
        
        
        drawn = draw(image, counts, objects, peaks, t)
        if camera.out:
            camera.out.write(drawn)
        cv2.imshow('flab2ab',drawn)
        cv2.waitKey(1)
        count += 1

    # Clean up resources
    print("Cleaning up")
    cv2.destroyAllWindows()
    camera.out.release()
    camera.cap.release()




if __name__ == "__main__":
    main()
