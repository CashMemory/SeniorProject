import argparse
import json
import os.path
import time

import cv2
import PIL.Image
import torch
import torch2trt
import torchvision.transforms as transforms
from torch2trt import TRTModule

import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects


class Model:
    """Deep learning model"""

    def __init__(self, pose_annotations, model_name):
        self.the_model = None
        self.model_trt = None
        self.height = None
        self.width = None
        self.model_weights = None
        self.optimized_model = None
        self.num_parts = len(pose_annotations["keypoints"])
        self.num_links = len(pose_annotations["skeleton"])

    def load_model(self, name):
        """Loads the TensorRT-optimized model which has been pre-trained on
        the MSCOCO dataset.

        :param name: A string indicating which model to use
        """
        # ------------- Constructing the model -------------------------
        # Each model takes at least two parameters, cmap_channels and
        # paf_channels corresponding to the number of heatmap channels and part
        # affinity field channels.
        #
        # The number of part affinity field channels is 2x the number of links,
        # because each link has a channel corresponding to the x and y
        # direction of the vector field for each link.
        if name == "resnet":
            print("------ model = resnet--------")
            # resnet18 was trained on an input resolution of 224x224
            self.width = 224
            self.height = 224
            self.model_weights = (
                "resnet18_baseline_att_224x224_A_epoch_249.pth"
            )
            self.optimized_model = (
                "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
            )
            self.the_model = (
                trt_pose.models.resnet18_baseline_att(
                    self.num_parts, 2 * self.num_links
                )
                .cuda()
                .eval()
            )
        else:
            print("------ model = densenet--------")
            # densenet121 was trained on an input resolution of 256x256
            self.width = 256
            self.height = 256
            self.model_weights = (
                "densenet121_baseline_att_256x256_b_epoch_160.pth"
            )
            self.optimized_model = (
                "densenet121_baseline_att_256x256_B_epoch_160_trt.pth"
            )
            self.the_model = (
                trt_pose.models.densenet121_baseline_att(
                    self.num_parts, 2 * self.num_links
                )
                .cuda()
                .eval()
            )

    def load_weights(self):
        """Load the model weights.

        Uses two loading functions
        1) torch.load: Deserializes pickled object files to memory and
           facilitates the device to load the data into
        2) load_state_dict: Loads a model's parameter dictionary using a
           deserialized state_dict
        (See https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        """
        # Load the model weights
        self.the_model.load_state_dict(torch.load(self.model_weights))

    def get_optimized(self):
        """Optimize this model by converting Pytorch to TensorRT."""
        # Optimize and save results if it's not already optimized
        if not os.path.exists(self.optimized_model):
            # Create sample data used to optimize with TensorRT
            self.data = torch.zeros((1, 3, self.height, self.width)).cuda()
            self.the_model.load_state_dict(torch.load(self.model_weights()))
            model_trt = torch2trt.torch2trt(
                self.the_model,
                [self.data],
                fp16_mode=True,  # use reduced half precision
                max_workspace_size=1 << 25,
            )
            torch.save(model_trt.state_dict(), self.optimized_model)

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(self.optimized_model))

    def log_fps(self):
        t0 = time.time()
        torch.cuda.current_stream().synchronize()
        for i in range(50):
            _ = self.model_trt(self.data)  # TODO not sure why this necessary
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        print("FPS log:", 50.0 / (t1 - t0))

    def execute_neural_net(self, data, parser):
        """Execute the neural network and parse the objects from its output

        :param data:
        :param parser:
        :return:
        """
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        #  cmap_threshold=0.15, link_threshold=0.15)
        counts, objects, peaks = parser(cmap, paf)
        return counts, objects, peaks


class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.out = None

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


# Image constants
WIDTH = 640
HEIGHT = 480
X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

# Image processing constants
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device("cuda")


def main():

    parser = argparse.ArgumentParser(description="TensorRT pose estimation")
    parser.add_argument("--model", type=str, default="resnet")
    args = parser.parse_args()

    # Load the annotation file and create a topology tensor
    with open("human_pose.json", "r") as f:
        human_pose = json.load(f)

    # Create a topology tensor (intermediate DS that describes part linkages)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    # Construct and load the model
    model = Model(pose_annotations=human_pose)
    model.load_model(args.model)
    model.load_weights()
    model.get_optimized()
    model.log_fps()

    # Set up the camera
    camera = Camera(width=WIDTH, height=HEIGHT)
    camera.capture_video("MP4V", "/tmp/output.mp4")  # OPTIONAL
    assert camera.cap is not None, "Camera Open Error"

    # Set up callable class used to parse the objects from the neural network
    parse_objects = ParseObjects(topology)  # from trt_pose.parse_objects

    # Execute while the camera is open and we haven't reached the time limit
    count = 0
    time_limit = 500
    while camera.cap.isOpened() and count < time_limit:
        t = time.time()
        succeeded, image = camera.cap.read()
        if not succeeded:
            print("Camera read Error")
            break

        resized_img = cv2.resize(
            image, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
        )
        preprocessed = preprocess(resized_img)
        counts, objects, peaks = model.execute_neural_net(
            data=preprocessed, parser=parse_objects
        )
        drawn = draw(resized_img, counts, objects, peaks, t)
        if camera.out:
            camera.out(drawn)
        count += 1

    # Clean up resources
    cv2.destroyAllWindows()
    camera.out.release()
    camera.cap.release()


def preprocess(image):
    """Preprocess an image, which is given in BGR8 / HWC format

    NOTE: mean, std, and device are pulled from global scope
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)  # move to cuda
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def get_keypoint(humans, hnum, peaks):
    """

    If a particular keypoint is found, it returns a coordinate value between
    0.0 and 1.0. Multiply this coordinate by the image size to calculate the
    exact location on the input image

    :param humans:
    :param hnum: A 0 based human index
    :param peaks:
    :return:
    """
    # check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]  # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
        else:
            peak = (j, None, None)
            kpoint.append(peak)
    return kpoint


def draw(src, counts, objects, peaks, time):
    color = (0, 255, 0)  # green
    fps = 1.0 / (time.time() - time)
    count = int(counts[0])
    for i in range(count):
        keypoints = get_keypoint(humans=objects, hnum=i, peaks=peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                cv2.circle(src, (x, y), 3, color, 2)
                cv2.putText(
                    src,
                    str(int(keypoints[j][0])),
                    (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    1,
                )
                cv2.circle(src, (x, y), 3, color, 2)
    cv2.putText(
        src,
        f"FPS: {fps}",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        1,
    )
    return src
