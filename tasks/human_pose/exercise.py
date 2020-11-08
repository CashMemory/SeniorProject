from helper import get_keypoint, WIDTH, HEIGHT, X_compress, Y_compress
import cv2
import time
import math

def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0] - p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0] - p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1.0

    try:
        value = math.acos((a+b-c) / math.sqrt(4*a*b)) * 180 / math.pi
        return value
    except ValueError as e:
        print("p0: ", p0, "p1: ", p1, "p2: ", p2)

class Exercise:

    def __init__(self, joints, angles, top_angle, middle_angle, bottom_angle,x_dist=0,y_dist=0):
        self.joints = [] #set of joints relevant to the exercise
        self.angles = [] #predetermined thresholds relevant to the exercise
        self.top_angle = 0
        self.middle_angle = 0
        self.bottom_angle = 0
        self.x_dist = 0
        self.y_dist = 0
        self.rep_count = 0
        self.rep_stack = []
        
class LeftBicepCurl():

    def __init__(self):
        self.joints = [5,7,9]
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
     
    def draw(self, src, counts, objects, peaks, t):
        print("!!!!! Bicep Drawing !!!!!")
        xy_dat = []
        color = (0, 255, 0)
        fps = 1.0 / (time.time() - t)
        has_data = True
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat.append((x,y))
                    print("Circles on joints")
                    cv2.circle(src, (x, y), 3, color, 2)
                    cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                    cv2.circle(src, (x, y), 3, color, 2)
            for point in xy_dat:
                if point is None:
                        print("xy_dat is incomplete!")
                        has_data = False
            if has_data and len(xy_dat) == 3:
                print(xy_dat)
                angle = angle_between_points(xy_dat[0], xy_dat[1], xy_dat[2])
                                
                # Red
                if angle < 60:
                    if self.rep_stack[-1] == "blue": 
                        self.rep_stack.append("red")
                    for data_point in xy_dat:
                        cv2.circle(img=src, 
                                center=data_point, 
                                radius=3, 
                                color=(255, 0, 0), 
                                thickness=2)
                # Blue
                elif angle > 100:
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []
                    for data_point in xy_dat:
                        cv2.circle(img=src, 
                                center=data_point, 
                                radius=3, 
                                color=(0, 0, 255), 
                                thickness=2)

        cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(src , "Rep: %d" % (self.rep_count), (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        return src

        

       
