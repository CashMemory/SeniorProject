class Exercise:

    def __init__(self, joints, angles, top_angle, middle_angle, bottom_angle,x_dist=0,y_dist=0):
        self.joints = []
        self.angles = []
        self.top_angle = 0
        self.middle_angle = 0
        self.bottom_angle = 0
        self.x_dist = 0
        self.y_dist = 0
        self.rep_count = 0
        
class LeftBicepCurl(Exercise):

    def __init__(self):
        super().__init__(self,[5,7,9],[150,90,15],15,90,150)

        

       