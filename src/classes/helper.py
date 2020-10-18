'''
This file holds miscelanous helper functions.


'''



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