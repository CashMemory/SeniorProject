import math
import time
from collections import namedtuple

import cv2

from helper import HEIGHT, WIDTH, X_compress, Y_compress, get_keypoint


def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    if a * b == 0:
        return -1

    try:
        value = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
        return int(value)
    except ValueError:
        print("p0: ", p0, "p1: ", p1, "p2: ", p2)


def distance_between_points(p0, p1):
    return int(math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2))


# BGR constants for drawing
BLUE = (194, 191, 33)
RED = (92, 62, 244)
GREEN = (160, 247, 9)

Function = namedtuple("Function", ["name", "args"])


class LeftBicepCurl:
    def __init__(self):
        self.name = "Left Curl"
        self.joints = [5, 7, 9]  # left shoulder, left elbow, left wrist
        self.angles = [150, 90, 15]
        self.rep_count = 0
        self.rep_stack = []
        self.rotations = [1] * 30
        self.arms = [0] * 30
        self.counter = 0
        self.last_known = None

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
        xy_dat = {}
        has_data = True
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat[j] = (int(x), int(y))
                    if j == 9:
                        cv2.circle(src, (x, y), 3, GREEN, 2)

            has_data = all(point in xy_dat for point in self.joints)

            if has_data:
                # angle calculation between lshoulder, lelbow, lwrist
                wrist_angle = 180 - angle_between_points(
                    xy_dat[5], xy_dat[7], xy_dat[9]
                )

                # forearm sampling
                forearm_length = distance_between_points(xy_dat[7], xy_dat[9])
                self.arms[self.counter % len(self.arms)] = forearm_length
                self.counter += 1
                forearm_length = self.get_arm_length()
                if forearm_length == 0:
                    continue

                # Define range-of-motion thresholds
                bottom_angle = 120
                top_angle = 50

                if (-1 * wrist_angle + 180) < top_angle:
                    if self.rep_stack and self.rep_stack[-1] == "blue":
                        self.rep_stack.append("red")

                elif (-1 * wrist_angle + 180) > bottom_angle:
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []

                rotation = 90
                # Save positions of arcs for drawing
                self.last_known = (
                    Function(
                        cv2.ellipse,
                        [
                            xy_dat[7],
                            (forearm_length, int(0.8 * forearm_length)),
                            rotation,
                            wrist_angle,
                            150,
                            RED,
                            4,
                        ],
                    ),
                    Function(
                        cv2.ellipse,
                        [
                            xy_dat[7],
                            (forearm_length, int(0.8 * forearm_length)),
                            rotation,
                            30,
                            wrist_angle,
                            BLUE,
                            4,
                        ],
                    ),
                )

            # Draw the arcs
            if self.last_known:
                for func in self.last_known:
                    func.name(src, *func.args)

        return src


class RightBicepCurl:
    def __init__(self):
        self.name = "Right Curl"
        self.joints = [
            6,
            8,
            10,
        ]  # Right Shoulder, Elbow, and Wrist
        self.angles = [150, 90, 15]
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
        xy_dat = {}
        has_data = True

        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            # check to see if data exists for each relevant keypoint
            # Then collect and draw
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat[j] = (int(x), int(y))
                    if j == 10:
                        cv2.circle(src, (x, y), 3, GREEN, 2)

            # checking for malformed xy data on relevant joints
            has_data = all(point in xy_dat for point in self.joints)

            # if data exists, we can do calculations
            if has_data:

                # Calculate angle between rshoulder, relbow, rwrist
                wrist_angle = angle_between_points(xy_dat[6], xy_dat[8], xy_dat[10])

                # forearm sampling
                forearm_length = distance_between_points(xy_dat[6], xy_dat[8])
                self.arms[self.counter % len(self.arms)] = forearm_length
                self.counter += 1
                forearm_length = self.get_arm_length()
                if forearm_length == 0:
                    continue

                # Define range-of-motion thresholds
                bottom_angle = 120
                top_angle = 50

                if wrist_angle < top_angle:
                    if self.rep_stack and self.rep_stack[-1] == "blue":
                        self.rep_stack.append("red")

                elif wrist_angle > bottom_angle:
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []

                rotation = -90

                # Save positions of arcs for drawing
                self.last_known = (
                    Function(
                        cv2.ellipse,
                        [
                            xy_dat[8],
                            (forearm_length, int(0.8 * forearm_length)),
                            rotation,
                            wrist_angle,
                            150,
                            BLUE,
                            4,
                        ],
                    ),
                    Function(
                        cv2.ellipse,
                        [
                            xy_dat[8],
                            (forearm_length, int(0.8 * forearm_length)),
                            rotation,
                            30,
                            wrist_angle,
                            RED,
                            4,
                        ],
                    ),
                )

            # Draw the arcs
            if self.last_known:
                for func in self.last_known:
                    print(func.args)
                    func.name(src, *func.args)

        return src


class ShoulderPress:
    def __init__(self):
        self.name = "Shoulder Press"
        self.joints = [5, 6, 7, 8, 9, 10, 17]  # LS, RS, LE, RE, LW, RW, NECK
        self.angles = [150, 90, 15]
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
        xy_dat = {}  # make dict
        has_data = True

        # find and draw keypoint locations
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            # collect all xy data for relevant joints and draw on joints
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    xy_dat[j] = (x, y)  # add xy data to dictionary
                    if j in [7, 8]:
                        cv2.circle(src, (x, y), 3, GREEN, 2)

            # see if data collected
            has_data = all(point in xy_dat for point in self.joints)

            if has_data:
                # find neck location
                neck_xy = xy_dat[17]
                # find length of one arm
                lshoulder_xy = xy_dat[5]
                lelbow_xy = xy_dat[7]
                arm_length = distance_between_points(lshoulder_xy, lelbow_xy)
                self.arms[self.counter % len(self.arms)] = arm_length
                self.counter += 1
                arm_length = self.get_arm_length()
                if arm_length == 0:
                    continue

                # create plane some % of arm length above neck and some lower plane
                # y values start at 0 in left corner -- grow larger as you move *DOWN*
                top_threshold = int(neck_xy[1] - (0.5 * arm_length))
                # Bottom threshold for elbow is just neck
                bottom_threshold = int(neck_xy[1])
                # check to see if both wrists are above plane

                # top of your range of motion
                if xy_dat[7][1] < top_threshold and xy_dat[8][1] < top_threshold:
                    if len(self.rep_stack) == 0:
                        continue
                    if self.rep_stack[-1] == "blue":
                        self.rep_stack.append("red")

                # bottom of your range of motion
                elif (
                    xy_dat[7][1] > bottom_threshold and xy_dat[8][1] > bottom_threshold
                ):
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("blue")
                    elif self.rep_stack[-1] == "red":
                        self.rep_count += 1
                        self.rep_stack = []

                if not self.rep_stack or self.rep_stack[-1] == "blue":
                    # Two lines with a break for your head
                    # Save last known lines for drawing
                    self.last_known = (
                        Function(
                            cv2.line,
                            [
                                (int(xy_dat[9][0]) - 32, top_threshold),
                                (int(xy_dat[9][0]) + 32, top_threshold),
                                RED,
                                2,
                            ],
                        ),
                        Function(
                            cv2.line,
                            [
                                (int(xy_dat[10][0]) - 32, top_threshold),
                                (int(xy_dat[10][0]) + 32, top_threshold),
                                RED,
                                2,
                            ],
                        ),
                    )
                else:
                    self.last_known = (
                        Function(
                            cv2.line,
                            [
                                (int(xy_dat[9][0]) - 32, bottom_threshold),
                                (int(xy_dat[9][0]) + 32, bottom_threshold),
                                BLUE,
                                2,
                            ],
                        ),
                        Function(
                            cv2.line,
                            [
                                (int(xy_dat[10][0]) - 32, bottom_threshold),
                                (int(xy_dat[10][0]) + 32, bottom_threshold),
                                BLUE,
                                2,
                            ],
                        ),
                    )

            if self.last_known:
                # Draw our last known planes (Helpful when we data incomplete)
                for func in self.last_known:
                    func.name(src, *func.args)

        return src


class Squat:
    def __init__(self):
        self.name = "Squat"
        self.joints = [11, 13]  # LH, LK
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
        xy_dat = {}  # make dict
        has_data = True

        # find and draw keypoint locations
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            # collect all xy data for relevant joints and draw on joints
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    # xy_dat.append((x,y))
                    xy_dat[j] = (x, y)  # add xy data to dictionary
                    if j == 11:  # Only draw the hip joint
                        cv2.circle(src, (x, y), 3, GREEN, 2)

            # see if data collected
            has_data = all(point in xy_dat for point in self.joints)

            if has_data:

                # find length of left thigh
                lhip_xy = xy_dat[11]
                lknee_xy = xy_dat[13]

                thigh_length = distance_between_points(lknee_xy, lhip_xy)
                self.thighs[self.counter % len(self.thighs)] = thigh_length
                self.counter += 1
                thigh_length = self.get_thigh_length()
                if thigh_length == 0:
                    continue

                # create plane some % of thigh length above knee and some lower plane
                # y values start at 0 in left corner -- grow larger as you move *DOWN*
                top_threshold = int(lknee_xy[1] - (0.75 * thigh_length))

                # Bottom threshold for elbow is just the knee
                bottom_threshold = int(lknee_xy[1] - (0.25 * thigh_length))
                # check to see if both wrists are above plane

                # top of your range of motion
                if xy_dat[11][1] < top_threshold:
                    print("!!!!! Red !!!!!")
                    if len(self.rep_stack) == 0:
                        self.rep_stack.append("red")
                    elif self.rep_stack[-1] == "blue":
                        self.rep_count += 1
                        self.rep_stack = []

                # bottom of your range of motion
                elif xy_dat[11][1] > bottom_threshold:
                    print("!!!!! Blue !!!!!")
                    if len(self.rep_stack) == 0:
                        continue
                    elif self.rep_stack[-1] == "red":
                        self.rep_stack.append("blue")

                # Save our line to draw
                if not self.rep_stack or self.rep_stack[-1] == "blue":
                    self.last_known = Function(
                        cv2.line, [(64, top_threshold), (576, top_threshold), RED, 2]
                    )
                else:
                    self.last_known = Function(
                        cv2.line,
                        [(64, bottom_threshold), (576, bottom_threshold), BLUE, 2],
                    )

            # Draw our last known plane
            if self.last_known:
                self.last_known.name(src, *self.last_known.args)

        return src


class Debug:
    def __init__(self):
        self.name = "Debug"
        self.rep_count = 0
        self.joints = list(range(18))
        self.skeleton = [
            [16, 14],  # Right ankle -> Right knee
            [14, 12],  # Right knee -> Right hip
            [15, 13],  # Left ankle -> Left knee
            [13, 11],  # Left knee -> Left hip
            [6, 8],  # Right elbow -> Right shoulder
            [8, 10],  # Right elbow -> Right wrist
            [7, 9],  # Left elbow -> Left wrist
            [5, 7],  # Left shoulder -> Left elbow
            [1, 3],  # Left eye -> Left ear
            [1, 0],  # Left eye -> nose
            [2, 4],  # Right eye -> Right ear
            [2, 0],  # Right eye -> nose
            [17, 0],  # Neck -> Nose
            [17, 5],  # Neck -> Left shoulder
            [17, 6],  # Neck -> Right shoulder
            [17, 12],  # Neck -> Right hip
            [17, 11],  # Neck -> Left hip
        ]

    def draw(self, src, counts, objects, peaks, t):
        fps = 1.0 / (time.time() - t)

        # find and draw keypoint locations
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)

            # collect all xy data for relevant joints and draw on joints
            for j in self.joints:
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)

                    print("Circles on joints")
                    cv2.circle(src, (x, y), 3, GREEN, 2)
                    cv2.putText(
                        src, f"{j}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 1
                    )

        cv2.putText(
            src, "FPS: %f" % (fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2
        )
        return src
