import PySimpleGUI as sg 
  


#cv MUST be passed in as a cv2 as cv.VideoCapture(0)
def MediaPlayerGUI(cv):
    background = '#27D796'
    elements = '#21BFC2'
    sg.set_options(background_color = background,element_background_color = elements)

    layout =[
                [sg.Text('Flab2Ab', size=(40, 1), justification='center', font='Helvetica 20')],
                [sg.Image(filename='', key='image')],
            ]
    
    window = sg.Window('Flab2Ab',location=(800,400))

    layout = [video_viewer_column]
        
    window = sg.Window("Flab2Ab",layout,default_element_size=(20,1),font=("Helcetica",25))

    cap = cv
    while True:
        event, values = window.ReadNonBlocking()

        ret, frame = cap.read()

        bio = io.Bytres() # binary memory resident stream
        img.save(bio, format= 'PNG')  # save image as png to it
        imgbytes = bio.getvalue()  # this can be used by OpenCV hopefully
        window.FindElement('image').Update(data=imgbytes)

MediaPlayerGUI(cv)













