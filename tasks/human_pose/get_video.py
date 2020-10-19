import argparse
import json
import os.path
import time

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




# Image constants
#WIDTH = 224
#$HEIGHT = 224
#X_compress = 640.0 / WIDTH * 1.0
#Y_compress = 480.0 / HEIGHT * 1.0

# Image processing constants
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device("cuda")


def main():

    print("Beginning script")
    parser = argparse.ArgumentParser(description="TensorRT pose estimation")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--output", type=str, default="/tmp/output.mp4")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    # Load the annotation file and create a topology tensor
    with open("human_pose.json", "r") as f:
        human_pose = json.load(f)

    # Create a topology tensor (intermediate DS that describes part linkages)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    
    # Construct and load the model
    model = Model(pose_annotations=human_pose)
    model.load_model(args.model)
    #model.load_weights()
    model.get_optimized()
    model.log_fps()
    print("Set up model")

    # Set up the camera
    camera = Camera(width=640, height=480)
    camera.capture_video("mp4v", args.output)
    assert camera.cap is not None, "Camera Open Error"
    print("Set up camera")

    # Set up callable class used to parse the objects from the neural network
    parse_objects = ParseObjects(topology)  # from trt_pose.parse_objects
    #draw_objects = DrawObjects(topology)  # from trt_pose.draw_objects

    print("Executing...")
    # Execute while the camera is open and we haven't reached the time limit
    count = 0
    time_limit = args.limit
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
