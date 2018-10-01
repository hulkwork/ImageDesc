import subprocess
import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))


def download_wget(url, output=os.path.join(dir_path, "../data/")):
    cmd = ["wget", "--directory-prefix=%s" % output,
           url]
    print(" ".join(cmd))
    subprocess.call(" ".join(cmd), shell=True)


def detect(c):
    """
    https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
    :param c:
    :return:
    """
    # initialize the shape name and approximate the contour
    shape = None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # return the name of the shape
    return shape


def dl_file(file_path, url):
    if not os.path.isfile(file_path):
        download_wget(url, os.path.dirname(file_path))

    if not os.path.isfile(file_path):
        raise IOError("please download %s from '%s' and put in %s" % (
        os.path.basename(file_path), url, os.path.dirname(file_path)))


def get_data():
    cascade_file = os.path.join(dir_path, "../data/cascade/lbpcascade_animeface.xml")
    url = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'
    dl_file(cascade_file,url)

    yolov3_cfg = os.path.join(dir_path, "../data/yolo/yolov3.cfg")
    url_yolov3_cfg = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
    dl_file(yolov3_cfg, url_yolov3_cfg)

    yolov3_txt = os.path.join(dir_path, "../data/yolo/yolov3.txt")
    url_yolov3_txt = 'https://raw.githubusercontent.com/arunponnusamy/object-detection-opencv/master/yolov3.txt'
    dl_file(yolov3_txt, url_yolov3_txt)

    yolov3_weights = os.path.join(dir_path, "../data/yolo/yolov3.weights")
    url_yolov3_weights = 'https://pjreddie.com/media/files/yolov3.weights'
    dl_file(yolov3_weights, url_yolov3_weights)
