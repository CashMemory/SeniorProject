# Bruno Capuano 2020
# display the camera feed using OpenCV
# display the camera feed with grayscale using OpenCV

import time
import cv2
import PySimpleGUI as sg

# Camera Settings
camera_Width  = 320 # 480 # 640 # 1024 # 1280
camera_Heigth = 240 # 320 # 480 # 780  # 960
frameSize = (camera_Width, camera_Heigth)
video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

# init Windows Manager
sg.theme("DarkBlue")

# def webcam col
colwebcam1_layout = [[sg.Text("Camera View", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

colwebcam2_layout = [[sg.Text("Camera View GrayScale", size=(60, 1), justification="center")],
                        [sg.Image(filename="", key="cam1gray")]]
colwebcam2 = sg.Column(colwebcam2_layout, element_justification='center')
colslayout = [colwebcam1, colwebcam2]

rowfooter = [sg.Image(filename="", key="-IMAGEBOTTOM-")]
layout = [colslayout, rowfooter]

window    = sg.Window("El Bruno – Webcams and GrayScale with PySimpleGUI", layout, 
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
    imgbytes = cv2.imencode(".png", gray)[1].tobytes()
    window["cam1gray"].update(data=imgbytes)

video_capture.release()
cv2.destroyAllWindows()