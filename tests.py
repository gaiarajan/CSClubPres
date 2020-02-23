
import os
import cv2
import numpy as np

from ARtest import *


IMG_DIR = "inputs"
VID_DIR = "inputs"
OUT_DIR = "output"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def helper(video_name, fps, output_name, is_part5):

    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = next(image_gen)
    h, w, d = image.shape

    out_path = "output/" + output_name
    video_out = mp4_video_writer(out_path, (w, h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))
    advert = cv2.imread(os.path.join(IMG_DIR, "advert.png"))
    src_points = get_corners_list(advert)

    frame_num = 1

    markers = None

    while image is not None:

        print ("Processing frame " + str(frame_num))

        markers = find_markers(image, template, markers)

        homography = find_four_point_transform(src_points, markers)
        image = project_imageA_onto_imageB(advert, image, homography)

        else:

            for marker in markers:
                mark_location(image, marker)

        video_out.write(image)

        image = next(image_gen)

        frame_num += 1

    video_out.release()


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)

def go():

    video_file = "vid1.mp4"
    fps = 40

    helper(video_file, fps, "5_a_1", True)

    video_file = "vid2.mp4"
    fps = 40

    helper(video_file, fps, "5_a_2", True)


if __name__ == '__main__':
    go() 
   
