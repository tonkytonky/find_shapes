from collections import OrderedDict
import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, contour):
        # initialize the shape name and approximate the contour
        perimeter = cv2.arcLength(curve=contour, closed=True)
        approx = cv2.approxPolyDP(
            curve=contour,
            epsilon=0.02 * perimeter,
            closed=True
        )

        # if the shape has 3 vertices, it is a triangle
        if len(approx) == 3:
            shape = 'triangle'

        # if the shape has 4 vertices, it is a rectangle
        elif len(approx) == 4:
            shape = 'rectangle'

        # otherwise, assume the shape is a circle
        else:
            shape = 'circle'

        # return the name of the shape
        return shape


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255)
        })

        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros(shape=(len(colors), 1, 3), dtype='uint8')
        self.color_names = []

        # loop over the colors dictionary
        for (color_number, (color_name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[color_number] = rgb
            self.color_names.append(color_name)

        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(src=self.lab, code=cv2.COLOR_RGB2LAB)

    def label(self, image, contour):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(shape=image.shape[:2], dtype='uint8')
        cv2.drawContours(
            image=mask,
            contours=[contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        # initialize the minimum distance found thus far
        min_dist = (np.inf, None)

        # loop over the known L*a*b* color values
        for (lab_num, color) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            distance = dist.euclidean(color[0], mean)

            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if distance < min_dist[0]:
                min_dist = (distance, lab_num)

        # return the name of the color with the smallest distance
        return self.color_names[min_dist[1]]


def _main():
    # load image and define template for counting figures
    image = cv2.imread(r'input.png')
    count_dict = OrderedDict((
        ('red rectangle', 0),
        ('blue triangle', 0),
        ('green circle', 0),
    ))

    # prepossess image
    blurred = cv2.GaussianBlur(
        src=image,
        ksize=(5, 5),
        sigmaX=0
    )
    gray = cv2.cvtColor(src=blurred, code=cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        src=gray,
        thresh=200,
        maxval=255,
        type=cv2.THRESH_BINARY_INV
    )[1]
    lab = cv2.cvtColor(src=blurred, code=cv2.COLOR_BGR2LAB)
    # show_img(thresh)

    # find contours in the thresholded image
    contours = cv2.findContours(
        image=thresh.copy(),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if imutils.is_cv2() else contours[1]

    # initialize the shape detector and color labeler
    shape_detector = ShapeDetector()
    color_labeler = ColorLabeler()

    # loop over the contours
    for contour in contours:
        # compute the center of the contour
        M = cv2.moments(contour)
        centerX = int(M["m10"] / M["m00"])
        centerY = int(M["m01"] / M["m00"])

        # detect the shape of the contour and label the color
        shape = shape_detector.detect(contour)
        color = color_labeler.label(lab, contour)

        # draw the contours and the name of the shape and labeled
        # color on the image
        contour_label = '{} {}'.format(color, shape)
        cv2.drawContours(
            image=image,
            contours=[contour],
            contourIdx=-1,
            color=(255, 0, 255),  # magenta
            thickness=2
        )
        cv2.putText(
            img=image,
            text=contour_label,
            org=(centerX - 20, centerY - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 255),  # magenta
            thickness=2
        )

        # show_img(image)
        count_contour(shape, color, count_dict)

    print_dictionary(count_dict)


def show_img(img):
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', img)
    cv2.waitKey(0)  # press any key to continue


def count_contour(shape, color, count_dict):
    try:
        count_dict['{} {}'.format(color, shape)] += 1
    except KeyError:
        pass


def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(key, value)


if __name__ == '__main__':
    _main()
