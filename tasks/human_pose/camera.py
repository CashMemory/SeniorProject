import cv2

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.out = None
        self.frame = None #stores the drawn frame

    def capture_video(self, codec, output_loc):
        """
        :param codec: String representing the video codec (use MP4V)
        :param output_loc: Location of the output file
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(
            output_loc,
            fourcc,
            self.cap.get(cv2.CAP_PROP_FPS),
            (self.width, self.height)
        )
