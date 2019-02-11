import argparse
import cv2
import numpy as np

from setup_logger import logger

HSV_BOUNDARIES = {
    "yellow": ([23, 50, 50], [35, 255, 255]),
    "green": ([69, 50, 50], [90, 255, 255]),
    "blue": ([90, 50, 50], [150, 255, 255]),
    "red": ([0, 50, 50], [10, 255, 255]),
    "black": ([0, 0, 0], [200, 255, 50]),
    "purple": ([150, 0, 0], [175, 255, 255])
}


class Picture():

    def __init__(self, filename, normalize=True):
        # RGB
        self.img_rgb = cv2.imread(filename=filename)
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

    def extract_color(self, colorname, boundaries):
        logger.info("Extracting the {} pathway".format(colorname))
        lower, upper = boundaries

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries
        # and apply the mask
        mask = cv2.inRange(self.img_hsv, lower, upper)
        # Erode a bit to remove small areas
        mask = cv2.erode(mask, None, iterations=2)
        # Dilate a bit to compensate bad detection
        mask = cv2.dilate(mask, None, iterations=3)

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
        cv2.imwrite(filename, np.hstack([self.img_rgb, extracted_path]))


def main(filename):
    picture = Picture(filename=filename)

    for colorname, boundaries in HSV_BOUNDARIES.iteritems():
        picture.extract_color(colorname, boundaries)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image")
    args = vars(ap.parse_args())

    main(filename=args["image"])
