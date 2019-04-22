import argparse
import cv2
import numpy as np

from background_removers import remove_background_canny
from background_removers import remove_background_hed
from setup_logger import logger

HSV_BOUNDARIES = {
    "red": ([0, 50, 50], [5, 255, 255]),
    "yellow": ([16, 50, 50], [30, 255, 255]),
    "green": ([40, 50, 50], [80, 255, 255]),
    "blue": ([90, 50, 50], [120, 255, 255]),
    "purple": ([120, 0, 0], [175, 255, 255]),
    "black": ([0, 0, 0], [360, 80, 150]),
}

RED_UPPER_BOUNDARIES = ([178, 50, 50], [180, 255, 255])


class Picture():

    def __init__(self, filename, normalize=True, debug=False):
        # RGB
        self.img_rgb = cv2.imread(filename=filename)
        self.bkgrd_mask = np.ones(self.img_rgb.shape[:2], dtype="uint8") * 255
        self.debug = debug
        if normalize:
            self.img_rgb = cv2.normalize(
                src=self.img_rgb,
                dst=self.img_rgb,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX
            )

        # Grayscale
        self.img_gray = cv2.cvtColor(
            cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )

        # HSV
        self.img_hsv = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2HSV)

    def remove_background(self, edge_detector):
        if edge_detector == "canny":
            self.bkgrd_mask = remove_background_canny(self.img_rgb ,self.debug)

        if edge_detector == "hed":
            self.bkgrd_mask = remove_background_hed(self.img_rgb ,self.debug)

    def extract_color(self, colorname, boundaries):
        logger.info("Extracting the {} pathway".format(colorname))
        lower, upper = boundaries

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries
        # and apply the mask
        mask_color = cv2.inRange(
            self.img_hsv,
            lower,
            upper)

        if colorname == "red":
            lower, upper = RED_UPPER_BOUNDARIES

            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask_color_neg = cv2.inRange(
                self.img_hsv,
                lower,
                upper)
            mask_color = cv2.bitwise_or(
                mask_color,
                mask_color_neg)


        if self.debug:
            cv2.imshow("Color mask", mask_color)
            cv2.waitKey(0)

        mask = cv2.bitwise_and(mask_color, self.bkgrd_mask)

        # Arbitraty number of pixel minimum to validate the color
        if sum(mask.flatten()) < 1000:
            logger.warning("Less than 1000 pixel for {}.".format(colorname))
            return

        # Extract colored part
        colored_part = cv2.bitwise_and(
            src1=self.img_rgb,
            src2=self.img_rgb,
            mask=mask)

        # Add blur to the gray version
        gray_image_blur = cv2.GaussianBlur(
            src=self.img_gray,
            ksize=(11, 11),
            sigmaX=0)

        # Extract the gray part
        gray_part = cv2.bitwise_and(
            src1=gray_image_blur,
            src2=gray_image_blur,
            mask=cv2.bitwise_not(mask))

        # Mix the gray part and the color part
        extracted_path = colored_part + gray_part

        # save the extracted_path
        filename = ''.join("{}_{}.jpg".format(
            args["image"].split(".")[0],
            colorname))

        # Side by side original picture and the extracted path
        if self.debug:
            cv2.imwrite(filename, np.hstack([self.img_rgb, extracted_path]))
        else:
            cv2.imwrite(filename, extracted_path)


def main(filename, edge_detector):
    picture = Picture(filename=filename, debug=False)
    if edge_detector:
        picture.remove_background(edge_detector)

    for colorname, boundaries in HSV_BOUNDARIES.items():
        picture.extract_color(colorname, boundaries)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
    ap.add_argument("-e", "--edge-detector", help="canny/hed")
    args = vars(ap.parse_args())

    main(filename=args["image"], edge_detector=args["edge_detector"])
