import cv2
import numpy as np
import os

from crop_layer import CropLayer
from setup_logger import logger


def remove_background_hed(img_rgb, debug):
    """
    https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
    """
    # load our serialized edge detector from disk
    print("[INFO] loading edge detector...")
    proto_path = os.path.sep.join([
        "hed_model",
        "deploy.prototxt"])
    model_path = os.path.sep.join([
        "hed_model",
        "hed_pretrained_bsds.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # register our new layer with the model
    cv2.dnn_registerLayer("Crop", CropLayer)

    # load the input image and grab its dimensions
    (H, W) = img_rgb.shape[:2]

    # construct a blob out of the input image for the Holistically-Nested
    # Edge Detector
    blob = cv2.dnn.blobFromImage(
        img_rgb,
        scalefactor=1.0,
        size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False)

    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    logger.info("performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    # Sharpen the hed mask
    hed[hed < 40] = 0

    contours, hierarchy = cv2.findContours(
        image=hed,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE)

    mask_shape = np.zeros(img_rgb.shape[:2], dtype="uint8")
    cv2.drawContours(
        image=mask_shape,
        contours=contours,
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED)

    filtered_img = cv2.bitwise_and(
        img_rgb,
        img_rgb,
        mask=mask_shape)

    if debug:
        cv2.imshow("HED", hed)
        cv2.imshow("Bkgrd Mask", filtered_img)
        cv2.waitKey(0)

    return mask_shape

def remove_background_canny(img_rgb, debug):

    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    if False:
        # Apply Gaussian Blur to remove small variation
        img = cv2.GaussianBlur(
            src=img,
            ksize=(3, 3),
            sigmaX=0)

    edges = auto_canny(img, sigma=1)

    contours, hierarchy = cv2.findContours(
        image=edges,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE)

    # Close all the shapes found
    new_contours = []
    for i, cnt in enumerate(contours):
        conv_cnt = cv2.convexHull(cnt)
        # Remove dot of the wall (number of pixel should be more than XXX)
        if cv2.contourArea(conv_cnt) > 10:
            new_contours.append(conv_cnt)

    mask_shape = np.ones(img.shape[:2], dtype="uint8") * 0
    cv2.drawContours(
        image=mask_shape,
        contours=new_contours,
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED)

    filtered_img = cv2.bitwise_and(
        img_rgb,
        img_rgb,
        mask=mask_shape)

    if debug:
        cv2.imshow("Canny", edges)
        cv2.imshow("Bkgrd Mask", filtered_img)
        cv2.waitKey(0)

    return mask_shape

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image[0])

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(
        image=image,
        threshold1=lower,
        threshold2=upper
    )
    # return the edged image
    return edged
