from helper import get_keypoint, WIDTH, HEIGHT, X_compress, Y_compress
import cv2
import time
import math
from collections import namedtuple

def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0] - p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0] - p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1

    try:
        value = math.acos((a+b-c) / math.sqrt(4*a*b)) * 180 / math.pi
        return int(value)
    except ValueError as e:
        print("p0: ", p0, "p1: ", p1, "p2: ", p2)

def distance_between_points(p0, p1):
    return int(math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2))

# BGR constants for drawing
BLUE = (194, 191, 33) 
RED = (92, 62, 244)
GREEN = (160, 247, 9)

Function = namedtuple('Function', ['name', 'args'])
        
class LeftBicepCurl():
    def __init__(self):
        self.joints = [5,7,9,11] #left shoulder, left elbow, left wrist, left hip
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
        self.rotations = [1] * 30
        self.arms = [0] * 30
        self.counter = 0
     
    def get_arm_length(self):
        # Check for a complete sampling of arm lengths
        if self.arms[-1] != 0:
            self.arms.sort()
            return int(self.arms[15])  # Get the median
        else:
            return 0

    def get_rotation(self):
        # Check for a complete sampling of arm lengths
        if self.arms[-1] != 1:
            self.arms.sort()
            return int(self.arms[15])  # Get the median
        else:
            return 1

    def draw(self, src, counts, objects, peaks, t):
        print("!!!!!Left Bicep Drawing !!!!!")
        xy_dat = {}
        #fps = 1.0 / (time.time() - t)
        has_data = True
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat[j] = (x,y)
                    if j == 9:
                        cv2.circle(src, (x, y), 3, GREEN, 2)

            has_data = all(point in xy_dat for point in self.joints)

            if has_data:
                #angle calculations
                wrist_angle = angle_between_points(xy_dat[5], xy_dat[7], xy_dat[9]) #angle between lshoulder, lelbow, lwrist
                
                #rotation sampling
                rotation = -1 * angle_between_points(xy_dat[11], xy_dat[7], xy_dat[9]) #angle between rhip, rshoulder, relbow
                self.rotations[self.counter % len(self.rotations)] = rotation
                rotation = self.get_rotation()
                if rotation == 1:
                    continue

                #forearm sampling
                forearm_length = distance_between_points(xy_dat[7], xy_dat[9])
                self.arms[self.counter % len(self.arms)] = forearm_length
                self.counter += 1
                forearm_length = self.get_arm_length()
                if forearm_length == 0:
                    continue  
                              
                bottom_angle = 120
                top_angle = 40     
                                
                if wrist_angle < top_angle:
                    if self.rep_stack[-1] == "blue": 
                        self.rep_stack.append("red")

                # Blue
                elif wrist_angle > bottom_angle:
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []

                # Save positions of arcs for drawing
                self.last_known = (Function(cv2.ellipse, [xy_dat[7], (forearm_length, int(.8 * forearm_length)), rotation, wrist_angle+5,bottom_angle, BLUE, 2]), 
                                   Function(cv2.ellipse, [xy_dat[7], (forearm_length, int(.8 * forearm_length)), rotation, top_angle, wrist_angle-5, 165, RED, 2]))
                                   
            # Draw the arcs
            if self.last_known:
                for func in self.last_known:
                    func.name(src, *func.args)

        return src

class RightBicepCurl():
    def __init__(self):
        self.joints = [6,8,10,12]  #Right Shoulder, Right Elbow, Right Wrist, right hip
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
        self.last_known = None
        self.arms = [0] * 30
        self.rotations = [-1] * 30
        self.counter = 0

    def get_arm_length(self):
        # Check for a complete sampling of arm lengths
        if self.arms[-1] != 0:
            self.arms.sort()
            return int(self.arms[15])  # Get the median
        else:
            return 0

    def get_rotation(self):
        # Check for a complete sampling of arm lengths
        if self.arms[-1] != -1:
            self.arms.sort()
            return int(self.arms[15])  # Get the median
        else:
            return -1

    def draw(self, src, counts, objects, peaks, t):
        print("!!!!!Right Bicep Drawing !!!!!")
        xy_dat = {}
        #fps = 1.0 / (time.time() - t)
        has_data = True

        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            #check to see if data exists for each relevant keypoint, collect and draw
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat[j] = (x,y)
                    if j == 10:
                        cv2.circle(src, (x, y), 3, GREEN, 2)

            #checking for malformed xy data on relevant joints
            has_data = all(point in xy_dat for point in self.joints)

            #if data exists, we can do calculations
            if has_data:
                print(xy_dat)

                #angle calculations
                wrist_angle = angle_between_points(xy_dat[6], xy_dat[8], xy_dat[10]) #angle between rshoulder, relbow, rwrist
                
                #rotation sampling
                rotation = angle_between_points(xy_dat[12], xy_dat[6], xy_dat[8]) #angle between rhip, rshoulder, relbow
                self.rotations[self.counter % len(self.rotations)] = rotation
                rotation = self.get_rotation()
                if rotation == -1:
                    continue

                #forearm sampling
                forearm_length = distance_between_points(xy_dat[6], xy_dat[8])
                self.arms[self.counter % len(self.arms)] = forearm_length
                self.counter += 1
                forearm_length = self.get_arm_length()
                if forearm_length == 0:
                    continue                
                bottom_angle = 120
                top_angle = 40        

                # Red
                if wrist_angle < top_angle:
                    if self.rep_stack[-1] == "blue": 
                        self.rep_stack.append("red")
                    #for data_point in xy_dat:
                        #cv2.circle(img=src,center=data_point, radius=3, color=(255, 0, 0), thickness=2)
                        
                # Blue
                elif wrist_angle > bottom_angle:
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []
                    #for data_point in xy_dat:
                        #cv2.circle(img=src,center=data_point,radius=3,color=(0, 0, 255),thickness=2)

                # Save positions of arcs for drawing
                self.last_known = (Function(cv2.ellipse, [xy_dat[8], (forearm_length, int(.8 * forearm_length)), rotation, wrist_angle+5,bottom_angle, BLUE, 2]), 
                                   Function(cv2.ellipse, [xy_dat[8], (forearm_length, int(.8 * forearm_length)), rotation, top_angle, wrist_angle-5, 165, RED, 2]))
                                   
            # Draw the arcs
            if self.last_known:
                for func in self.last_known:
                    func.name(src, *func.args)
            

        #cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        #cv2.putText(src , "Rep: %d" % (self.rep_count), (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        return src

class ShoulderPress():
    def __init__(self):
        self.joints = [5,6,7,8,9,10,17] #LS, RS, LE, RE, LW, RW, NECK
        self.angles = [150,90,15]
        self.rep_count = 0
        self.rep_stack = []
        self.arms = [0] * 30
        self.counter = 0  # Counting samples for arm length
        self.last_known = None

    def get_arm_length(self):
        # Check for a complete sampling of arm lengths
        if self.arms[-1] != 0:
            self.arms.sort()
            return int(self.arms[15])  # Get the median
        else:
            return 0
    
    def draw(self, src, counts, objects, peaks, t):
        print("!!!!! SHOULDER PRESS !!!!!")
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
                    xy_dat[j] = (x, y) #add xy data to dictionary
                    if j in [7, 8]:
                        print("Circles on joints")
                        cv2.circle(src, (x, y), 3, GREEN, 2)
            
            #see if data collected
            has_data = all(point in xy_dat for point in self.joints)

            if has_data:
                #find neck location
                neck_xy = xy_dat[17]
                #find length of one arm
                lshoulder_xy = xy_dat[5]
                lelbow_xy = xy_dat[7]
                arm_length = distance_between_points(lshoulder_xy,lelbow_xy) 
                self.arms[self.counter % len(self.arms)] = arm_length
                self.counter += 1
                arm_length = self.get_arm_length()
                if arm_length == 0:
                    continue

                #create plane some % of arm length above neck and some lower plane
                # y values start at 0 in left corner -- grow larger as you move *DOWN*
                top_threshold = int(neck_xy[1] - (.5 * arm_length))
                # Bottom threshold for elbow is just neck
                bottom_threshold = int(neck_xy[1]) 
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
                    # for point in self.joints:
                    #     print("Painting red")
                    #     if point in [7, 8]:  # Only draw elbows
                    #         cv2.circle(img=src, center=xy_dat[point], radius=3, color=(0, 0, 255), thickness=2)

                # Blue -- (bottom of your range of motion)
                elif xy_dat[7][1] > bottom_threshold and xy_dat[8][1] > bottom_threshold:
                    print("!!!!! Blue !!!!!")
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []
                    #for point in self.joints:
                        #if point in [7,8]:  # Only draw elbows
                            #cv2.circle(img=src, center=xy_dat[point], radius=3, color=(255, 0, 0), thickness=2)
        
                if not self.rep_stack or self.rep_stack[-1] == "blue":
                    # Two lines with a break for your head
                    # Save last known lines for drawing
                    self.last_known = (Function(cv2.line, [(128, top_threshold), (neck_xy[0] - 32, bottom_threshold), RED, 2]), 
                                        Function(cv2.line, [(neck_xy[0] + 32, top_threshold), (512, top_threshold), RED, 2]))
                    #self.last_known = (cv2.line(src, (128, int(top_threshold)), (neck_xy[0] - 32, int(top_threshold)), color=RED, thickness=2),
                    #                   cv2.line(src, (neck_xy[0] + 32, int(top_threshold)), (512, int(top_threshold)), color=RED, thickness=2))
                else:
                    self.last_known = (Function(cv2.line, [(128, bottom_threshold), (neck_xy[0] - 32, bottom_threshold), BLUE, 2]), 
                                       Function(cv2.line, [(neck_xy[0] + 32, bottom_threshold), (512, bottom_threshold), BLUE, 2]))
                    # self.last_known = (cv2.line(src, (128, int(bottom_threshold)), (neck_xy[0] - 32, int(bottom_threshold)), color=BLUE, thickness=2),
                    #                    cv2.line(src, (neck_xy[0] + 32, int(bottom_threshold)), (512, int(bottom_threshold)), color=BLUE, thickness=2))
            
            if self.last_known:
                # Draw our last known planes (Helpful when we have incomplete data)
                for func in self.last_known:
                    func.name(src, *func.args)

        
        return src

class Squat():
    def __init__(self):
        self.joints = [11,13]  #LH, LK
        self.rep_count = 0
        self.rep_stack = []
        self.thighs = [0] * 30
        self.counter = 0
        self.last_known = None

    def get_thigh_length(self):
        # Check for a complete sampling of thigh lengths
        if self.thighs[-1] != 0:
            self.thighs.sort()
            return int(self.thighs[15])  # Get the median
        else:
            return 0

    def draw(self, src, counts, objects, peaks, t):
        print("!!!!! SQUAT !!!!!")
        xy_dat = {} #make dict
        #color = (0, 255, 0)
        #fps = 1.0 / (time.time() - t)
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
                    if j == 11:  # Only draw the hip joint
                        cv2.circle(src, (x, y), 3, GREEN, 2)
            
            #see if data collected
            has_data = all(point in xy_dat for point in self.joints)

            if has_data:
                
                #find length of left thigh
                lhip_xy = xy_dat[11]
                lknee_xy = xy_dat[13]
                
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
                #cv2.circle(img=src, center=(lhip_xy[0], int(top_threshold)), radius=8, color=(255, 0, 255), thickness=2)
                
                # Bottom threshold for elbow is just the knee
                bottom_threshold = int(lknee_xy[1] - (.25 * thigh_length))
                print("$$$$ Bottom:", bottom_threshold)
                #cv2.circle(img=src, center=(lhip_xy[0], int(bottom_threshold)), radius=8, color=(255, 0, 255), thickness=2)
                #check to see if both wrists are above plane

                #Red -- (top of your range of motion)
                if xy_dat[11][1] < top_threshold:
                    print("!!!!! Red !!!!!")
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("red")
                    elif self.rep_stack[-1] == "blue":
                        self.rep_count += 1
                        self.rep_stack = []
                    #for point in self.joints:
                        #print("Painting red")
                        #if point == 11:  # Only draw knee joint
                            #cv2.circle(img=src, center=xy_dat[point], radius=3, color=(0, 0, 255), thickness=2)

                # Blue -- (bottom of your range of motion)
                elif xy_dat[11][1] > bottom_threshold:
                    print("!!!!! Blue !!!!!")
                    if len(self.rep_stack) == 0:
                        continue
                    elif self.rep_stack[-1] == "red":
                        self.rep_stack.append("blue")
                    #for point in self.joints:
                        #if point == 11:  # Only draw knee joint
                            #cv2.circle(img=src, center=xy_dat[point], radius=3, color=(255, 0, 0), thickness=2)

                # Save our line to draw
                if not self.rep_stack or self.rep_stack[-1] == "blue":
                    self.last_known = Function(cv2.line, [(64, top_threshold), (576, top_threshold), RED, 2])
                    # self.last_known = cv2.line(src, (64, top_threshold), (576, top_threshold), RED, 2)
                else:
                    #self.last_known = cv2.line(src, (64, bottom_threshold), (576, bottom_threshold), BLUE, 2)
                    self.last_known = Function(cv2.line, [(64, bottom_threshold), (576, bottom_threshold), BLUE, 2])
                    
            # Draw our last known plane
            if self.last_known:
                # FIXME -- Could be syntax issues here. Also, are we drawing on an old frame (src) ? 
                self.last_known.name(src, self.last_known.args)
        
        #cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        #cv2.putText(src , "Rep: %d" % (self.rep_count), (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        return src

class Debug():
    def __init__(self):
        self.joints = list(range(18))
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 8], [7, 9], [8, 10], [9, 11], 
                        [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [18, 1], [18, 6], [18, 7], 
                        [18, 12], [18, 13]]

    def draw(self, src, counts, objects, peaks, t):
        print("!!!!! DEBUG !!!!!")
        xy_dat = {} #make dict
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
                    cv2.circle(src, (x, y), 3, GREEN, 2)
                    cv2.putText(src, f"{j}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 1) 
            

            #iterate over the skeleton, which is list of of pairs of connections [lw, le], [le, ls], ls, neck]
            for joint in self.skeleton:
                a = xy_dat[joint[0]]
                b = xy_dat[joint[1]]
                cv2.line(src, a, b, BLUE, 1)
                

        cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 1)


# From NVIDIA ... remove this if the above function works for debug mode
    # def __call__(self, image, object_counts, objects, normalized_peaks):
    #     topology = self.topology
    #     height = image.shape[0]
    #     width = image.shape[1]
        
    #     K = topology.shape[0]
    #     count = int(object_counts[0])
    #     K = topology.shape[0]
    #     for i in range(count):
    #         color = (0, 255, 0)
    #         obj = objects[0][i]
    #         C = obj.shape[0]
    #         for j in range(C):
    #             k = int(obj[j])
    #             if k >= 0:
    #                 peak = normalized_peaks[0][j][k]
    #                 x = round(float(peak[1]) * width)
    #                 y = round(float(peak[0]) * height)
    #                 cv2.circle(image, (x, y), 3, color, 2)

    #         for k in range(K):
    #             c_a = topology[k][2]
    #             c_b = topology[k][3]
    #             if obj[c_a] >= 0 and obj[c_b] >= 0:
    #                 peak0 = normalized_peaks[0][c_a][obj[c_a]]
    #                 peak1 = normalized_peaks[0][c_b][obj[c_b]]
    #                 x0 = round(float(peak0[1]) * width)
    #                 y0 = round(float(peak0[0]) * height)
    #                 x1 = round(float(peak1[1]) * width)
    #                 y1 = round(float(peak1[0]) * height)
    #                 cv2.line(image, (x0, y0), (x1, y1), color, 2)