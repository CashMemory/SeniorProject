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

def distance_between_points(p0, p1):
    return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)


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
        self.joints = [5,7,9] #left shoulder, left elbow, left wrist
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
     
    def draw(self, src, counts, objects, peaks, t):
        print("!!!!!Left Bicep Drawing !!!!!")
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
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("red")
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

class RightBicepCurl():
    def __init__(self):
        self.joints = [6,8,10]  #Right Shoulder, Right Elbow, Right Wrist
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []

    def draw(self, src, counts, objects, peaks, t):
        print("!!!!!Right Bicep Drawing !!!!!")
        xy_dat = []
        color = (0, 255, 0)
        fps = 1.0 / (time.time() - t)
        has_data = True

        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            #check to see if data exists for each relevant keypoint, collect and draw
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat.append((x,y))
                    print("Circles on joints")
                    cv2.circle(src, (x, y), 3, color, 2)
                    cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                    cv2.circle(src, (x, y), 3, color, 2)

            #checking for malformed xy data on relevant joints
            for point in xy_dat:
                if point is None:
                        print("xy_dat is incomplete!")
                        has_data = False

            #if data exists, we can do calculations
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

class ShoulderPress():
    def __init__(self):
        #TODO: determine joints, angles distance
        self.joints = [5,6,7,8,9,10,17] #LS, RS, LE, RE, LW, RW, NECK
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
        self.arms = [0] * 30
        self.counter = 0  # Counting samples for arm length

    def get_arm_length(self):
        # Check for a complete sampling of arm lengths
        if self.arms[-1] != 0:
            self.arms.sort()
            return self.arms[15]  # Get the median
        else:
            return 0
    
    def draw(self, src, counts, objects, peaks, t):
        #TODO: finish implementation
        print("!!!!! SHOULDER PRESS !!!!!")
        xy_dat = {} #make dict
        #xy_dat = [] #make list
        color = (0, 255, 0)
        fps = 1.0 / (time.time() - t)
        has_data = True
        #find and draw keypoint locations
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            #collect all xy data for relevant joints and draw on joints            
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    #xy_dat.append((x,y))
                    xy_dat[j] = (x, y) #add xy data to dictionary
                    print("Circles on joints")
                    cv2.circle(src, (x, y), 3, color, 2)
                    cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                    cv2.circle(src, (x, y), 3, color, 2)
            
            #see if data collected
            for point in self.joints:
                if xy_dat.get(point) is None:
                    print("xy_dat is incomplete!")
                    has_data = False

            if has_data:
                #find neck location
                neck_xy = xy_dat[17]
                #find length of one arm
                lshoulder_xy = xy_dat[5]
                lelbow_xy = xy_dat[7]
                lwrist_xy = xy_dat[9]
                #arm_length = distance_between_points(lshoulder_xy,lelbow_xy) + distance_between_points(lelbow_xy, lwrist_xy)
                arm_length = distance_between_points(lshoulder_xy,lelbow_xy) 
                self.arms[self.counter % len(self.arms)] = arm_length
                self.counter += 1
                arm_length = self.get_arm_length()
                if arm_length == 0:
                    continue

                #create plane some % of arm length above neck and some lower plane
                # y values start at 0 in left corner -- grow larger as you move *DOWN*
                top_threshold = neck_xy[1] - (.5 * arm_length)
                cv2.circle(img=src, center=(neck_xy[0], int(top_threshold)), radius=8, color=(255, 0, 255), thickness=2)
                #bottom_threshold = neck_xy[1] - (.15 * arm_length) -- TEMP -- wrist
                # Bottom threshold for elbow is just neck
                bottom_threshold = neck_xy[1] 
                #check to see if both wrists are above plane

                #TODO more sophisticated plane chekcing
                #Red -- (top of your range of motion)
                if xy_dat[7][1] < top_threshold and xy_dat[8][1] < top_threshold:
                    print("!!!!! Red !!!!!")
                    if len(self.rep_stack) == 0:
                        continue
                        #self.rep_stack.append("blue")
                    if self.rep_stack[-1] == "blue":
                        self.rep_stack.append("red")
                    for point in self.joints:
                        print("Painting red")
                        cv2.circle(img=src, 
                                center=xy_dat[point], 
                                radius=3, 
                                color=(0, 0, 255), 
                                thickness=2)

                # Blue -- (bottom of your range of motion)
                elif xy_dat[7][1] > bottom_threshold and xy_dat[8][1] > bottom_threshold:
                    print("!!!!! Blue !!!!!")
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []
                    for point in self.joints:
                        cv2.circle(img=src, 
                                center=xy_dat[point], 
                                radius=3, 
                                color=(255, 0, 0), 
                                thickness=2)
        
        cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(src , "Rep: %d" % (self.rep_count), (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                #invert
        return src

class Squat():
    def __init__(self):
        self.joints = [11,13,15]  #LH, RH, LK, RK, LA, RA
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
        self.thighs = [0] * 30
        self.counter = 0

    def get_thigh_length(self):
        # Check for a complete sampling of thigh lengths
        if self.thighs[-1] != 0:
            self.thighs.sort()
            return self.thighs[15]  # Get the median
        else:
            return 0

    def draw(self, src, counts, objects, peaks, t):
        print("!!!!! SQUAT !!!!!")
        xy_dat = {} #make dict
        color = (0, 255, 0)
        fps = 1.0 / (time.time() - t)
        has_data = True
        #find and draw keypoint locations
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            #collect all xy data for relevant joints and draw on joints            
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    #xy_dat.append((x,y))
                    xy_dat[j] = (x, y) #add xy data to dictionary
                    print("Circles on joints")
                    cv2.circle(src, (x, y), 3, color, 2)
                    cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                    cv2.circle(src, (x, y), 3, color, 2)
            
            #see if data collected
            for point in self.joints:
                if xy_dat.get(point) is None:
                    print("xy_dat is incomplete!")
                    has_data = False

            if has_data:
                
                #find length of left thigh
                lhip_xy = xy_dat[11]
                lknee_xy = xy_dat[13]
                lankle_xy = xy_dat[15]
                
                thigh_length = distance_between_points(lknee_xy,lhip_xy) 
                self.thighs[self.counter % len(self.thighs)] = thigh_length
                self.counter += 1
                thigh_length = self.get_thigh_length()
                if thigh_length == 0:
                    continue

                #create plane some % of thigh length above knee and some lower plane
                # y values start at 0 in left corner -- grow larger as you move *DOWN*
                top_threshold = int(lknee_xy[1] - (.75 * thigh_length))
                #TODO: switch statement for bottom and top for reps
                print("$$$$ Top:", top_threshold)
                cv2.line(src, (64, top_threshold), (576, top_threshold), (255, 0, 255), 2)
                #cv2.circle(img=src, center=(lhip_xy[0], int(top_threshold)), radius=8, color=(255, 0, 255), thickness=2)
                
                # Bottom threshold for elbow is just the knee
                bottom_threshold = int(lknee_xy[1] - (.25 * thigh_length))
                print("$$$$ Bottom:", bottom_threshold)
                cv2.line(src, (64, bottom_threshold), (576, bottom_threshold), (255, 0, 255), 2)
                #cv2.circle(img=src, center=(lhip_xy[0], int(bottom_threshold)), radius=8, color=(255, 0, 255), thickness=2)
                #check to see if both wrists are above plane

                #TODO more sophisticated plane chekcing
                #Red -- (top of your range of motion)
                if xy_dat[11][1] < top_threshold:
                    print("!!!!! Red !!!!!")
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("red")
                    elif self.rep_stack[-1] == "blue":
                        self.rep_count += 1
                        self.rep_stack = []
                    for point in self.joints:
                        print("Painting red")
                        cv2.circle(img=src, 
                                center=xy_dat[point], 
                                radius=3, 
                                color=(0, 0, 255), 
                                thickness=2)

                # Blue -- (bottom of your range of motion)
                elif xy_dat[11][1] > bottom_threshold:
                    print("!!!!! Blue !!!!!")
                    if len(self.rep_stack) == 0:
                        continue
                    elif self.rep_stack[-1] == "red":
                        self.rep_stack.append("blue")
                    for point in self.joints:
                        cv2.circle(img=src, 
                                center=xy_dat[point], 
                                radius=3, 
                                color=(255, 0, 0), 
                                thickness=2)
        
        cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(src , "Rep: %d" % (self.rep_count), (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        return src
