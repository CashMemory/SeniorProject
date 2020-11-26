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

from camera import Camera
from helper import draw, preprocess, WIDTH, HEIGHT
from model import Model
from exercise import LeftBicepCurl, RightBicepCurl, ShoulderPress, Squat, Debug

from flask import Flask, Response, render_template
from flask_restful import Api, Resource

import PySimpleGUI as sg

executing = False       #global flag for session start
exercise = None         #global exercise object required for model inference and drawing
stopExercise = False    #global flag for stopping exercise after loop ends
drawn = None            #global for our image

class LeftCurlAPI(Resource):
    def get(self):        
        global exercise
        exercise = LeftBicepCurl()
        global executing
        executing = True
        return {'leftCurl':f'{id}'}

class RightCurlAPI(Resource):
    def get(self):        
        global exercise
        exercise = RightBicepCurl()
        global executing
        executing = True
        return {'rightCurl':f'{id}'}

class ShoulderPressAPI(Resource):
    def get(self):        
        global exercise
        exercise = ShoulderPress()
        global executing
        executing = True
        return {'shoulderPress':f'{id}'}

class SquatAPI(Resource):
    def get(self):
        global exercise
        exercise = Squat()
        global executing
        executing = True
        return {'squat':f'{id}'}

class RepCountAPI(Resource):
    def get(self):  
        global exercise    
        reps = exercise.rep_count if exercise else 0
        return {'repCount':f'{reps}'}

class EndExerciseAPI(Resource):
    def get(self):
        global stopExercise
        stopExercise = True          
        return {'endExercise':f'{id}'}

class StartSessionAPI(Resource):
    def get(self):
        return {'startSession':f'{id}'}

class DebugAPI(Resource):
    def get(self):
        global exercise
        exercise = Debug()
        global executing
        executing = True
        return {'debug':f'{id}'}

# ------ Begin GUI layout ------
workout_list = ['workout1','workout2','workout3']

background = '#27D796'
element_bg = '#232530'
elements = '#1A1C23'


sg.set_options(background_color = background, text_color = elements )

# def webcam col
colwebcam1_layout = 
[   
    #[sg.Image(filename=r"logo.png")],
    [sg.Text("Camera Feed", background_color = background)],
    [sg.Image(filename="", key="cameraFeed")]
]

colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')


WorkoutList = 
[         
    [sg.Text("Rep Count", size=(60, 1), justification="center", background_color = background)],
    [sg.Text("1", size=(60, 1), justification="center",key="repcount", background_color = background)],
    [sg.Text("Workout List", size=(60, 1), justification="center", background_color = background)],
    [sg.Listbox(values=[],size=(60,30),enable_events=True, key="workoutlist")]
]

worklist = sg.Column(WorkoutList, element_justification='center')

layout = 
[
    [colwebcam1,sg.VSeperator(),worklist]
]


window    = sg.Window("FLAB2AB", layout, 
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False, 
                    return_keyboard_events=False, location=(100, 100), finalize=True)        




def main():

    print("Beginning script")
    # Load the annotation file and create a topology tensor
    with open("human_pose.json", "r") as f:
        human_pose = json.load(f)

    # Create a topology tensor (intermediate DS that describes part linkages)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    
    # # Construct and load the model
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

    #add endpoints
    api.add_resource(LeftCurlAPI, '/leftCurl')
    api.add_resource(RightCurlAPI, '/rightCurl')
    api.add_resource(ShoulderPressAPI, '/shoulderPress')
    api.add_resource(SquatAPI, '/squat')
    api.add_resource(RepCountAPI, '/repCount')
    api.add_resource(EndExerciseAPI, '/endExercise')
    api.add_resource(StartSessionAPI, '/startSession')
    api.add_resource(DebugAPI, '/debug')
    
    t = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0"})  # threaded=True)
    t.start()

    print("After networking")
    while not executing:
        pass
    
    print("Executing...")
    # Execute while the camera is open and we haven't reached the time limit

    global exercise, stopExercise, drawn

   

    while True:

        while camera.cap.isOpened() and exercise:
            t = time.time()

            #Gui Stuff
            event, values = window.read(timeout=1)

            if event == 'Exit' or event == sg.WIN_CLOSED:
                break;

            succeeded, image = camera.cap.read()

            print("Frame captured")
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
            
            
            drawn = exercise.draw(image, counts, objects, peaks, t)
            #drawn = draw(image, counts, objects, peaks, t)
            camera.frame = drawn

            if camera.out:
                camera.out.write(drawn)

            imgbytes = cv2.imencode(".png", drawn)[1].tobytes()
            window["cameraFeed"].update(data=imgbytes)     

            
            #cv2.imshow('flab2ab',drawn)
            #cv2.waitKey(1)
                   
            if stopExercise:
                exercise = None
                stopExercise = False
                search = values['workoutlist']
                new_values = [x for x in workout_list if search in x]
                window.Element('workoutlist').update(new_values)

                print("exercise ended successfully")

    # Clean up resources
    print("Cleaning up")
    cv2.destroyAllWindows()
    camera.out.release()
    camera.cap.release()

if __name__ == "__main__":
    main()
