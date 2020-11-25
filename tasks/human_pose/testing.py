import time
#   from cv2 import cv2
import PySimpleGUI as sg

def main():

    col = [[sg.Text('This is the first line')],
       [sg.In()],
       [sg.Button('Save'), sg.Button('Exit')]]

    layout = [[sg.Column(col, key='-COLUMN-')]]     # put entire layout into a column so it can be saved

    window = sg.Window("Drawing and Moving Stuff Around", layout)

    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break  # exit
        elif event == 'Save':
            filename = sg.popup_get_file('Choose file (PNG, JPG, GIF) to save to', save_as=True)
            save_element_as_file(window['-COLUMN-'], filename)

    window.close()