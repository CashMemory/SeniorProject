import time
import torch
import cv2


def preprocess(image):
    """Preprocess an image, which is given in BGR8 / HWC format

    NOTE: mean, std, and device are pulled from global scope
    """
    global device
    device = torch.device("cuda")
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
    :param humans: :param hnum: A 0 based human index
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

WIDTH = 224
HEIGHT = 224
X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0


def draw(src, counts, objects, peaks, t):
    color = (0, 255, 0)  # green
    fps = 1.0 / (time.time() - t)
    for i in range(counts[0]):
        keypoints = get_keypoint(humans=objects, hnum=i, peaks=peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                cv2.circle(src, (x, y), 3, color, 2)
                cv2.putText(src, "%d" % int(keypoints[j][0]), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.circle(src, (x, y), 3, color, 2)
    cv2.putText(
        src,
        "FPS: %d" % fps,
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        1
    )
    return src
