import os
import numpy as np
from absl import logging
from matplotlib import pyplot as plt
from cellpose import models
import cv2
import re


def makeFolder(folderName: str):
    """
    make folderName if not exist
    Parameters
    ----------
    folderName string

    Returns create non-existed folder
    -------

    """
    if not os.path.exists(folderName):
        os.mkdir(folderName)
        logging.info('Create folder {}'.format(folderName))
    else:
        # logging.info('folder {} exist'.format(folderName))
        pass


def fileLists(folder: str, delimiter="") -> list:
    """
    return list of the filtered delimited files in folder
    :param folder:
    :param delimiter:
    :return:
    """
    return sorted([x for x in os.listdir(folder) if x.endswith(delimiter) or x.startswith(delimiter)])
    # return sorted([x for x in os.listdir(folder)])


def rescale_image(imgIn: np.ndarray, bitDepth: int = 8) -> np.ndarray:
    """
    rescale img to 8bit image
    @param imgIn:
    @param bitDepth:
    @return: image array
    """
    rows, cols = imgIn.shape
    imgIn = np.double(imgIn)
    imgMax = np.max(imgIn)
    imgMin = np.min(imgIn)
    imgOut = np.zeros_like(imgIn)
    imgOut = (imgIn - imgMin) / (imgMax - imgMin) * ((2 ** bitDepth) - 1)
    return imgOut


def centroid(x: np.ndarray) -> np.double:
    """
    function to calculate the centroid of the signal
    @param x:
    @return:
    """
    logging.debug(f"numerator is {np.sum(x * (1 + np.arange(len(x))))}")
    logging.debug(f"denominator is {np.sum(x)}")
    return np.sum(x * (1 + np.arange(len(x)))) / np.sum(x)


def detect_cells(image, diameter=120, model=models.Cellpose(model_type='cyto')):
    channels = [0, 0]
    masks, _, _, _ = model.eval(image, diameter=diameter, channels=channels)
    return masks


def cell_info(masks: np.ndarray, img: np.ndarray) -> np.ndarray:
    """

    :param masks:
    :param img:
    :return:
    """
    num_cells = masks.max()
    locations = np.zeros((num_cells, 9))
    for i in range(1, num_cells + 1):
        mask = masks == i
        ymin, xmin = mask.nonzero()[0].min(), mask.nonzero()[1].min()
        ymax, xmax = mask.nonzero()[0].max(), mask.nonzero()[1].max()
        xMean = np.mean(mask, 0)
        yMean = np.mean(mask, 1)
        locations[i - 1, 0] = i
        locations[i - 1, 1] = centroid(xMean)
        locations[i - 1, 2] = centroid(yMean)
        locations[i - 1, 3] = np.sum(img * mask) / np.sum(mask) - np.min(np.ravel(img))
        locations[i - 1, 4] = np.min(np.ravel(img))
        locations[i - 1, 5] = xmin
        locations[i - 1, 6] = ymin
        locations[i - 1, 7] = xmax
        locations[i - 1, 8] = ymax
    return locations


def cell_detection(img: np.ndarray, gaussianKernel: tuple = (3, 3), imgMin: int = 0, imgMax: int = 255, boxX=30,
                   boxY=30):
    """
    :param img:
    :param gaussianKernel:
    :param imgMin:
    :param imgMax:
    :param boxX:
    :param boxY:
    :return:
    """

    # Gaussian blur to remove the hot/dark pixel
    blur = cv2.GaussianBlur(img, gaussianKernel, 0)

    # threshold
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    result = np.zeros((100, 4))
    for contour in contours:
        # Obtain the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # if w > boxX or h > boxY:
        if w > boxX and h > boxY:
            result[count, 0:6] = np.array([x, y, w, h])
            count += 1
    return np.uint16(result[:count - 1, :])


def peak_location(dataIn: np.ndarray) -> np.ndarray:
    return np.argmax(dataIn)


def peak_val(dataIn: np.ndarray) -> np.ndarray:
    return np.max(np.ravel(dataIn))


def full_width_half_maximum(dataIn: np.ndarray) -> tuple:
    """

    :param dataIn:
    :return:
    """

    # set default value
    idxMax = 0
    rightLinearFit = 0
    leftLinearFit = 0

    # index of max
    idxMax = np.argmax(dataIn)
    logging.debug("argmax is {}".format(idxMax))
    logging.debug("max is is {}".format(dataIn[idxMax]))

    # half maximum value
    halfMax = dataIn[idxMax] / 2
    logging.debug("half of max is is {}".format(halfMax))

    threshold = halfMax
    mask = dataIn >= threshold
    mask = mask.astype(int)
    firstIndex = (mask != 0).argmax(axis=0)

    rightSideData = dataIn[idxMax:]
    if np.sum(rightSideData <= threshold):
        secondIndex = rightSideData <= threshold
        secondIndex = secondIndex.astype(int)
        secondIndex = (secondIndex != 0).argmax(axis=0) + idxMax
    else:
        secondIndex = len(dataIn)

    if firstIndex == idxMax:
        leftLinearFit = idxMax
    elif (firstIndex < idxMax) & (firstIndex > 0):
        leftLinearFit = linear_fit(dataIn[firstIndex - 1:firstIndex + 1],
                                   np.arange(firstIndex - 1, firstIndex + 1, 1), threshold)
    else:
        leftLinearFit = 0
    logging.debug("leftLinearFit is {}".format(leftLinearFit))

    if secondIndex == len(dataIn):
        rightLinearFit = len(dataIn) - 1
    elif secondIndex < len(dataIn):
        rightLinearFit = linear_fit(dataIn[secondIndex - 2:secondIndex + 2],
                                    np.arange(secondIndex - 2, secondIndex + 2, 1), threshold)
    else:
        rightLinearFit = idxMax
    logging.debug("rightLinearFit is {}".format(rightLinearFit))

    return idxMax, dataIn[idxMax], rightLinearFit - leftLinearFit, leftLinearFit, rightLinearFit


def timing_energy(dataIn: np.ndarray, threshold: float = 1.1) -> tuple:
    """

    :param dataIn:
    :param threshold:
    :return:
    """

    mask = dataIn >= threshold
    threshDataIn = dataIn[mask]
    energy = np.sum(threshDataIn)

    # set default value
    idxMax = 0
    rightLinearFit = 0
    leftLinearFit = 0

    # index of max
    idxMax = np.argmax(dataIn)

    # set threshold
    mask = dataIn >= threshold
    mask = mask.astype(int)
    firstIndex = (mask != 0).argmax(axis=0)
    rightSideData = dataIn[idxMax:]

    if np.sum(rightSideData <= threshold):
        secondIndex = rightSideData <= threshold
        secondIndex = secondIndex.astype(int)
        secondIndex = (secondIndex != 0).argmax(axis=0) + idxMax
    else:
        secondIndex = len(dataIn)

    if firstIndex == idxMax:
        leftLinearFit = idxMax
    elif (firstIndex < idxMax) & (firstIndex > 0):
        leftLinearFit = linear_fit(dataIn[firstIndex - 1:firstIndex + 1],
                                   np.arange(firstIndex - 1, firstIndex + 1, 1), threshold)
    else:
        leftLinearFit = 0
    logging.debug("leftLinearFit is {}".format(leftLinearFit))

    if secondIndex == len(dataIn):
        rightLinearFit = len(dataIn) - 1
    elif secondIndex < len(dataIn):
        rightLinearFit = linear_fit(dataIn[secondIndex - 2:secondIndex + 2],
                                    np.arange(secondIndex - 2, secondIndex + 2, 1), threshold)
    else:
        rightLinearFit = idxMax
    logging.debug("rightLinearFit is {}".format(rightLinearFit))

    if idxMax - leftLinearFit == 0:
        riseTime = 1
    else:
        riseTime = idxMax - leftLinearFit
    fallTime = rightLinearFit - idxMax

    return energy, riseTime, fallTime, idxMax, leftLinearFit, rightLinearFit


def linear_fit(x, y, target):
    """
    linear fit (x, y) and return y=m*target+b
    :param x: np.ndarray
    :param y: np.ndarray
    :param target: np.ndarray
    :return: np.ndarray
    """
    p = np.polyfit(x, y, 1)
    return np.polyval(p, target)



# def time_stamp(file_name):
#     print(file_name)
    # pattern = r'(\d{2})h(\d{2})m_(\d{2})s'
    # match = re.search(pattern, file_name)
#     if match:
#         hour = int(match.group(1))
#         minute = int(match.group(2))
#         second = int(match.group(3))
#         return hour, minute, second
#     else:
#         return None

def time_stamp(file_name):
    print(file_name)
    pattern = r'^.*_(\d*)_.*$'
    match = re.search(pattern, file_name)
    if match:
        hour = 0
        minute = 0
        second = int(match.group(1))
        return hour, minute, second
    else:
        return None