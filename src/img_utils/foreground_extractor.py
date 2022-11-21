import numpy as np
import cv2
from .utils import *


def extract_foreground_contour(img: np.array, k_size=(5, 5)):
    blurred_img = cv2.GaussianBlur(img, k_size, 0)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray_img, 200, 255, cv2.CHAIN_APPROX_NONE)
    contour = find_largest_contour(gray)
    img_contour = np.copy(img)
    cv2.drawContours(img_contour, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [contour], 255)
    # create a copy of the current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
    res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
    res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels
    mask2 = np.where(
        (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')
    # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
    new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask3d = new_mask3d
    mask3d[new_mask3d > 0] = 255.0
    mask3d[mask3d > 255] = 255.0
    # apply Gaussian blurring to smoothen out the edges a bit
    # `mask3d` is the final foreground mask (not extracted foreground image)
    mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)
    # show('Foreground mask', mask3d)
    # create the foreground image by zeroing out the pixels where `mask2`...
    # ... has black pixels
    foreground = np.copy(img).astype(float)
    foreground[mask2 == 0] = 0
    return foreground


def extract_foreground_morphology(img: np.array, k_size=(16, 16)):
    hh, ww = img.shape[:2]
    
    # threshold on white colour
    lower = np.array([100, 100, 100])
    upper = np.array([255, 255, 255])

    # create mask to only select colour outside white
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize= k_size)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    return result