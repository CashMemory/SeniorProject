import torch
import torch2trt
import torchvision.transforms as transforms
from torch2trt import TRTModule

import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects

import time

class Model:
    """Deep learning model"""

    def __init__(self, pose_annotations):
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
        # Create sample data used to optimize with TensorRT
        self.data = torch.zeros((1, 3, self.height, self.width)).cuda()
        # Optimize and save results if it's not already optimized

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
