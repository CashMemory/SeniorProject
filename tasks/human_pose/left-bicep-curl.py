import math
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import numpy as np
import copy

def angle_between_points(p0, p1, p2):
    a = (p1[0] - p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0] - p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0] - p0[0])**2 + (p2[1]-p0[1])**2
    print("a: ", a, "b: ", b, "c: ", c)
    if a * b == 0:
        return -1.0

    try:
        value = math.acos((a+b-c) / math.sqrt(4*a*b)) * 180 / math.pi
        return value
    except ValueError as e:
        print("p0: ", p0, "p1: ", p1, "p2: ", p2)

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint


parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute(img, src, t):
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)
    xy_dat = []
    has_data = True
    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        left_bicep_curl = [5, 7, 9]
        for j in left_bicep_curl:
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                xy_dat.append((x,y))
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
            if angle < 60:
                for data_point in xy_dat:
                    cv2.circle(src, data_point, 3, (255, 0, 0), 2)
            elif angle > 100:
                for data_point in xy_dat:
                    cv2.circle(src, data_point, 3, (0, 0, 255), 2)
            print(f"Angle: {angle}")
    print("FPS:%f "%(fps))
    
    #draw_objects(img, counts, objects, peaks)

    cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    #resized = cv2.resize(src, (640, 480))
    #cv2.imshow('Pose Estimation', resized)
    out_video.write(src)
    return src



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
count = 0

X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
#draw_objects = DrawObjects(topology)

while cap.isOpened() and count < 200:
    t = time.time()
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break

    # im2 = np.copy(dst)
    # im3 = copy.deepcopy(dst)
    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    cv2.imshow('test', execute(img, dst, t))
    cv2.waitKey(1)
    count += 1


cv2.destroyAllWindows()
#out_video.release()
cap.release()



