# Bruno Capuano 2020
# display the camera feed using OpenCV
# display the camera feed with grayscale using OpenCV

import time
from cv2 import cv2
import PySimpleGUI as sg


workout_list = ['workout1','workout2','workout3']
# Camera Settings
camera_Width  = 320 # 480 # 640 # 1024 # 1280
camera_Heigth = 240 # 320 # 480 # 780  # 960
frameSize = (camera_Width, camera_Heigth)
video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

# init Windows Manager
background = '#232530'
elements = '#27D796'
sg.set_options(background_color = background,element_background_color = elements)

# def webcam col
colwebcam1_layout = [[sg.Text("Camera View", size=(100, 1), justification="center")],
                        [sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

WorkoutList = [[sg.Text("Workout List", size=(60, 1), justification="center")],
                        [sg.Listbox(workout_list, key="workoutlist")]]
worklist = sg.Column(WorkoutList, element_justification='center')
colslayout = [colwebcam1, worklist]

rowfooter = [sg.Image(filename="", key="-IMAGEBOTTOM-")]
layout = [colslayout, rowfooter]

window    = sg.Window("FLAB2AB", layout, 
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False, 
                    return_keyboard_events=True, location=(100, 100))        
while True:
    start_time = time.time()
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED:
        break

    # get camera frame
    ret, frameOrig = video_capture.read()
    frame = cv2.resize(frameOrig, frameSize)
  
    # if (time.time() – start_time ) > 0:
    #     fpsInfo = "FPS: " + str(1.0 / (time.time() – start_time)) # FPS = 1 / time to process loop
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, fpsInfo, (10, 20), font, 0.4, (255, 255, 255), 1)

    # # update webcam1
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["cam1"].update(data=imgbytes)
    
    # # transform frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # update webcam2
    search = values['workoutlist']
    new_values = [x for x in workout_list if search in x]
    window.Element('workoutlist').update(new_values)

video_capture.release()
cv2.destroyAllWindows()