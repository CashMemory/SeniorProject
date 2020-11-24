import PySimpleGUI as sg 
  


#cv MUST be passed in as a cv2 as cv.VideoCapture(0)
def MediaPlayerGUI():
    background = '#27D796'
    elements = '#21BFC2'
    sg.set_options(background_color = background,element_background_color = elements)

    layout =[
                [sg.Text('Flab2Ab', size=(40, 1), justification='center', font='Helvetica 20')], 
            ]
    
    window = sg.Window('Flab2Ab',location=(800,400))
        
    window = sg.Window("Flab2Ab",layout,default_element_size=(20,1),font=("Helcetica",25))


    while True:
        window["Flab2Ab"].update(data=encoded_img)

    

MediaPlayerGUI()













