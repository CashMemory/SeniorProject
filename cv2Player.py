from cv2 import cv2

def play_videoFile(filePath,mirror=False):
    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video Life2Coding',cv2.WINDOW_AUTOSIZE)
    while True:
        ret_val, frame = cap.read()
        if mirror:
            frame = cv2.flip(frame, 1)
        cv2.imshow('Video Life2Coding', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
    
def main():
    play_videoFile('bigbuckbunny720p5mb.mp4',mirror=False)


if __name__ == '__main__':
    main()