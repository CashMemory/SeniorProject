import cv2
import PySimpleGUI as sg
from camera import Camera
import numpy as np

layout = [[sg.Image(filename='', key="cameraFeed")]]

window = sg.Window("Flab2Ab", layout, location=(800,400), finalize=True)


def main():

    camera = Camera(width=640, height=480)
    camera.capture_video("mp4v", "/tmp/output.mp4")
    assert camera.cap is not None, "Camera Open Error"
    print("Set up camera")

    while True:

        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:

            break

        succeeded, image = camera.cap.read()

        print("Frame captured")
        if not succeeded:
            print("Camera read Error")
            break
    
        imgbytes = cv2.imencode(".png", image)[1].tobytes()

        window["cameraFeed"].update(data=imgbytes)
        

main()