import cv2
import numpy as np


def order_point(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # for the top left and bottom right points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # for the top right and bottom left diff will be min and max
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
def four_point_transform(image, pts):
    rect = order_point(pts)
    (tl, tr, br, bl) = rect

    # find maximun width possible
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(widthA,widthB)

    # find minimum width possible
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    maxWidth = int(maxWidth)
    maxHeight = int(maxHeight)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

