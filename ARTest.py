import cv2
import numpy as np
import math
import operator


def find_markers(image, template = None, prevFrameMarkers = None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.
        prevFrameMarkers:  List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
            that were the marker locations in the previous frame of the video

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    """average convolution result and prev frame"""
    print("running function for markers")
    copy = np.copy(image)
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    copy = cv2.GaussianBlur(image,(9,9),0)
    #find corners and get set of points
    harris = cv2.cornerHarris(gray, blockSize = 6, ksize = 7, k = 0.04)
    (r, c) = np.where(harris > np.max(harris) / 8.0)
    points = np.float32(np.vstack((c, r)).T)  #convert the output of np.where to a 2d array of points (Nx2), this is needed for using kmeans
    corner_copy = np.copy(image)
    for p in points:
        cv2.circle(corner_copy, tuple(p), 1, (234, 26, 232), thickness = -1)  #show points on the image, note: we need to change p from a list to a tuple
        compactness, classified_points, means = cv2.kmeans(data = points, K=4,
            bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
            attempts=1, flags=cv2.KMEANS_PP_CENTERS)
    ret = []
    for i in range(4):
        ret.append((int(means[i][0]), int(means[i][1])))
    print(ret)
    ret.sort(key = operator.itemgetter(0))
    ret[0:2] = sorted(ret[0:2], key=lambda tup: tup[1])
    print(ret)
    ret[2:4] = sorted(ret[2:4], key=lambda tup: tup[1])
    print(ret)
    return ret

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    cv2.line(image, markers[0], markers[1], thickness)
    cv2.line(image, markers[0], markers[2], thickness)
    cv2.line(image, markers[1], markers[3], thickness)
    cv2.line(image, markers[2], markers[3], thickness)
    return image



def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    h, mask = cv2.findHomography(np.float32(src_points), np.float32(dst_points), cv2.RANSAC,5.0)
    return h


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the homography given we should be able to easily project imageA into the correct spot in imageB
   
    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    height, width, channels = imageB.shape
    final = cv2.warpPerspective(imageA, homography, (width, height), imageB, borderMode=cv2.BORDER_TRANSPARENT)
    return final

def get_corners_list(image):
    """Returns a list of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    return [(0, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, 0), (image.shape[1] - 1, image.shape[0] - 1)];
