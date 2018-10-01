import os
from src import imagine
from utils import misc
from conf import config
import argparse
parser = argparse.ArgumentParser()
parser.parse_args()
parser.add_argument("--path", help="path to your image",
                    type=str)
parser.add_argument("-v","--verbose", help="print details",
                    type=int, default=1)

args = parser.parse_args()

dir_path = os.path.dirname(os.path.realpath(__file__))
misc.get_data()

if __name__ == "__main__":

    if args.path:
        desc = imagine.ImageDescription(image_path=args.path, conf=config.conf, verbose=args.verbose)
        desc.process()

    else:
        path_1 = os.path.join(dir_path, "data/naruto-shippuden.jpg")
        path_2 = os.path.join(dir_path, "data/narutosh.jpeg")
        path_3 = os.path.join(dir_path, "data/shape_detection_thresh.jpg")
        path_4 = os.path.join(dir_path, "data/messi.jpg")
        for p in [path_1, path_2, path_3, path_4]:
            desc = imagine.ImageDescription(image_path=p, conf=config.conf, verbose=args.verbose)
            desc.process()
