import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from absl import app
from absl import flags
from absl import logging
from cellpose import models, plot
import tools
import re
import json
from datetime import datetime, timedelta

FLAGS = flags.FLAGS
# flags.DEFINE_string("folder",
#                     r"C:\Users\sophi\OPALS\AIImageAnalysis\data\RGC\RGCWTfov_15_noshock\Default",
                    # "Folder name")
flags.DEFINE_integer("diameter", 120, "Custom diameter value (in pixels) used for the cellpose model")

flags.DEFINE_integer("mode", 0, "0-Cell_Search, 1-MaxCells, 2-Data_Analysis, 3-Post_Analysis")

def cell_search(folderName: str, diameter: int, experimentName: str):
    # global rois
    destFolder = os.path.join(os.getcwd(), experimentName)
    tools.makeFolder(destFolder)
    destFolder = os.path.join(destFolder, "output")
    tools.makeFolder(destFolder)
    destFolder = os.path.join(destFolder, "cell_search")
    tools.makeFolder(destFolder)
    logging.debug(folderName)
    fileNames = tools.fileLists(folderName, delimiter="tif")
    logging.debug(fileNames)

    logging.info("{} Start image analysis {}".format(50 * "=", 50 * "="))
    logging.info("{} detect the brightest frame {}".format(50 * "-", 44 * "-"))
    cyto_model = models.Cellpose(gpu = True, model_type="cyto")
    for idx, fileName in enumerate(fileNames[:]):
        if idx%100 == 0:
            temp = cv2.imread(os.path.join(folderName, fileName), -1)
            img = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            masks = tools.detect_cells(img, diameter=120, model=cyto_model)
            num_cells = masks.max()
            cellInfo = tools.cell_info(masks, temp)
            logging.info("idx->{},filename->{},numberOfCells->{}".format(idx, fileName, num_cells))
            np.savetxt(os.path.join(destFolder, "{}.csv".format(fileName[:-4])), cellInfo, delimiter=",",
                    header="index,x,y,signal,background,xmin,ymin,xmax,ymax", comments="")
    logging.info("{} Finish image analysis {}".format(49 * "=", 49 * "="))

def maximum_cells(folderName: str, experimentName: str):
    destFolder = os.path.join(os.path.join(os.path.join(folderName, experimentName), "output"), "cell_search")
    print("destination here " + destFolder)
    fileNames = tools.fileLists(destFolder, delimiter="csv")
    df = pd.DataFrame()
    for idx, fileName in enumerate(fileNames[:]):
        if idx % 50 == 0:
            logging.info("filename is {}".format(fileName[:-4]))
        temp = pd.read_csv(os.path.join(destFolder, fileName))
        temp["file"] = fileName[:-4]
        df = pd.concat([df, temp])

    columns = df.columns
    fileSns = df["file"].unique()
    cellCounts = np.zeros(len(fileSns), )
    for idx, fileSn in enumerate(fileSns):
        cellCounts[idx] = len(df[df["file"] == fileSn])
    maxIdx = np.argmax(cellCounts)
    with open(os.path.join(destFolder[:-12], "max_cell_frame.csv"), "w") as fp:
        fp.write("max_cell_frame\n")
        fp.write(str(maxIdx)+"\n")
    logging.info("the maximum cell happened at frame {}".format(maxIdx))

    plt.rcParams["font.size"] = "14"
    style = "seaborn-v0_8-darkgrid"
    plt.style.use(style)
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    ax.plot(cellCounts, "-")
    ax.plot(maxIdx, cellCounts[maxIdx], "-h", markersize=8)
    ax.axvline(maxIdx, color="r", linestyle=":", lw=1.5)
    ax.text(maxIdx + 5, cellCounts[maxIdx], str(maxIdx), color="r")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cell Count")
    plt.savefig(os.path.join(destFolder, "cell_count.png"), dpi=150)
    print(destFolder)
    plt.close()

def data_analysis(folderName: str, experimentFolder: str, experimentName: str):
    logging.debug(folderName)

    destFolder = os.path.join(folderName, "output")
    pictureFolder = os.path.join(experimentFolder, experimentName)
    tools.makeFolder(destFolder)

    maxFrameFileName = os.path.join(destFolder, "max_cell_frame.csv")
    maxFrame = np.int16(np.genfromtxt(maxFrameFileName, skip_header=1))
    logging.debug(maxFrame)

    time_stamps = []
    fileNames = tools.fileLists(folderName, delimiter="tif")
    today = datetime.now()
    prev_hr = 0
    prev_min = 0
    prev_sec = 0
    prev_hr, prev_min, prev_sec = tools.time_stamp(fileNames[0])
    # prev_time = datetime(year=today.year, month=today.month, day=today.day, hour=prev_hr, minute=prev_min, second=prev_sec)
    prev_time = prev_sec
    for file in fileNames: 
        curr_hr = 0
        curr_min = 0
        curr_sec = 0
        curr_hr, curr_min, curr_sec = tools.time_stamp(file)
        # curr_time = datetime(year=today.year, month=today.month, day=today.day, hour=curr_hr, minute=curr_min, second=curr_sec)
        curr_time = curr_sec
        timediff = curr_time - prev_time
        # time_stamps.append(timediff.total_seconds())
        time_stamps.append(timediff)
        prev_hr = curr_hr
        prev_min = curr_min
        prev_sec = curr_sec
    logging.info(time_stamps)
    logging.debug(fileNames)

    logging.info("{} Start image analysis {}".format(50 * "=", 50 * "="))
    logging.info("{} detect the brightest frame {}".format(50 * "-", 44 * "-"))

    fileName = fileNames[maxFrame]
    temp = cv2.imread(os.path.join(folderName, fileName), -1)
    img = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    masks = tools.detect_cells(img, diameter=120, model=models.Cellpose(gpu = True, model_type="cyto"))
    cellInfo = tools.cell_info(masks, temp)

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    plt.figure(figsize=(8, 8))
    plt.imshow(masks, cmap="gray")
    for idx, cell in enumerate(cellInfo):
        plt.text(cell[1], cell[2], "roi_{}".format(idx),  fontdict=font)
    plt.title(fileName)
    plt.tight_layout()
    plt.savefig(os.path.join(pictureFolder, "mask.png"), dpi=150)
    plt.close()
    num_cells = masks.max()
    result = pd.DataFrame()

    for idx, fileName in enumerate(fileNames[:]):
        if idx % 50 == 0:
            logging.info("{}".format(fileName))

        temp = cv2.imread(os.path.join(folderName, fileName), -1)
        background = temp.min()
        df = pd.DataFrame()
        df['index'] = [idx]
        df['time_stamp'] = time_stamps[idx]
        df['background'] = [background]

        for roiIdx in range(0, num_cells):
            logging.debug("{}".format(roiIdx))
            mask = (masks == roiIdx)
            mask = mask.astype(int)
            logging.debug("{}, {}".format(roiIdx, np.sum(mask)))
            filterImg = temp * mask
            df["roi_{}".format(roiIdx)] = [np.sum(filterImg)/np.sum(mask)]
        result = pd.concat([result, df])

    result.to_csv(os.path.join(destFolder, "final_brightness.csv"), index=False)
    logging.debug("\n{}".format(result))
    logging.info("{} Finish image analysis {}".format(49 * "=", 49 * "="))

def post_analysis(folderName: str, threshold: float, experimentFolder: str, experimentName: str):
    # destFolder = os.path.join(folderName, "roi_stats")
    pictureFolder = os.path.join(experimentFolder, experimentName)
    # tools.makeFolder(destFolder)
    plt.rcParams['font.size'] = '14'
    style = 'seaborn-v0_8-darkgrid'
    plt.style.use(style)
    fileName = os.path.join('output', 'final_brightness.csv')
    df = pd.read_csv(os.path.join(folderName, fileName))
    columns = df.columns
    logging.debug("columns are {}".format(columns))
    resultDf = pd.DataFrame()
    logging.info("{} Start image analysis {}".format(50 * "=", 50 * "="))
    time_stamps = df["time_stamp"]
    for column in columns[3:]:
        logging.info(column)
        dataIn = np.array(df[column])
        baseline = df[column] - df['background']
        dataIn = baseline / baseline[0]

        energy, riseTime, fallTime, maxLoc, left, right = tools.timing_energy(dataIn, threshold=threshold)
        before_intensities = dataIn[0:int(left)]
        average_intensities = np.mean(before_intensities)
        peakLocation, peakVal, fwhm, leftIdx, rightIdx = tools.full_width_half_maximum(dataIn)
        temp = pd.DataFrame()
        temp['roi'] = [column.split("_")[1]]
        temp['background'] = [df['background'].iloc[peakLocation]]
        temp['baseline'] = [baseline[0]]
        temp['threshold'] = [threshold]
        temp['peak'] = [peakVal]
        temp['peak_location'] = [peakLocation]
        temp['FWHM'] = [fwhm]
        temp['rise_time'] = [riseTime]
        temp['fall_time'] = [fallTime]
        temp['energy'] = [energy]
        temp['fwhm_left_index'] = [leftIdx]
        temp['fwhm_right_index'] = [rightIdx]
        temp['threshold_left_index'] = [left]
        temp['threshold_right_index'] = [right]
        temp['average_before_peak'] = [average_intensities]
        resultDf = pd.concat([resultDf, temp])

        # generate graph
        plt.figure(figsize=(8, 6))
        plt.plot(time_stamps, dataIn, '-', lw=2)
        plt.plot(peakLocation, peakVal, 'h', markersize=12)
        plt.plot(leftIdx, peakVal / 2, 'cx', markersize=12)
        plt.plot(rightIdx, peakVal / 2, 'mx', markersize=12)
        plt.axhline(y=peakVal / 2, color='r', linestyle=':', lw=1.5)
        plt.axhline(y=threshold, color='g', linestyle='-.', lw=1.5)

        plt.axvline(x=leftIdx, color='r', linestyle=':', lw=1.5)
        plt.axvline(x=rightIdx, color='r', linestyle=':', lw=1.5)
        plt.axvline(x=left, color='b', linestyle=':', lw=1.5)
        plt.axvline(x=right, color='b', linestyle=':', lw=1.5)
        plt.text(leftIdx + 20, peakVal * .55, "FWHM is {:.3f}".format(fwhm))
        plt.text(left + 10, threshold + .3, "Rise time is {:.3f}".format(riseTime))
        plt.text(len(dataIn) // 2, threshold + .1, "Fall time is {:.3f}".format(fallTime))
        plt.title("{}".format(column))
        plt.text(left // 2, threshold + .2, "Average Intensity Before Peak: {:.3f}".format(average_intensities))
        # plt.xlabel("Time (Seconds)")
        plt.xlabel("Frame #")
        plt.ylabel("Signal [DN]")
        plt.tight_layout()
        plt.savefig(os.path.join(pictureFolder, "statistics_{}.png".format(column)), dpi=150)
        plt.close()
    resultDf.to_csv(os.path.join(pictureFolder, "statistics.csv"), index=False)
    logging.info("{} Finish image analysis {}".format(49 * "=", 49 * "="))
    plt.show()

def main(argv):
    config_file = open('config.json', 'r')
    config = json.load(config_file)
    config_file.close()
    folderName = config['folder_location']
    diameter = FLAGS.diameter
    mode = FLAGS.mode
    os.makedirs(os.path.join(os.getcwd(), config['experiment_name']), exist_ok=True)
    if mode == 0:
        cell_search(folderName, diameter, config['experiment_name'])
    if mode <= 1:
        maximum_cells(os.getcwd(), config['experiment_name'])
    if mode <= 2:
        data_analysis(folderName, os.getcwd(), config['experiment_name'])
    if mode <= 3:
        post_analysis(folderName, 1.1, os.getcwd(), config['experiment_name'])
    # try:
    #     cell_search(folderName, diameter)
    # except:
    #     logging.error("CELL SEARCH FAILED")

    # try:
    #     maximum_cells(folderName)
    # except:
    #     logging.error("MAXIMUM CELLS FAILED")

    # try:
    #     data_analysis(folderName)
    # except:
    #     logging.error("DATA ANALYSIS FAILED")

    # try:
    #     post_analysis(folderName, 1.1)
    # except:
    #     logging.error("POST ANALYSIS FAILED")

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
