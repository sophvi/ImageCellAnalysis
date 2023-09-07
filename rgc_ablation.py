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
from datetime import datetime, timedelta

FLAGS = flags.FLAGS
flags.DEFINE_string("folder",
                    r"C:\Users\sophi\OPALS\AIImageAnalysis\data\RGC\RGCWTfov_15_noshock\Default",
                    "Folder name")
flags.DEFINE_integer("diameter", 120, "Custom diameter value (in pixels) used for the cellpose model")

flags.DEFINE_integer("mode", 0, "0-Square_Mask" "1-Data_Analysis, 2-Post_Analysis")

# flags.DEFINE_integer("mode", 0, "0-Intensity_Search" "1-Cell_Search, 2-MaxCells, 3-Data_Analysis, 4-Post_Analysis")

def intensity_search(folderName: str):
    # destFolder = os.path.join(folderName, "intensities")
    # tools.makeFolder(destFolder)
    # logging.debug(folderName)
    file_names = tools.fileLists(folderName, delimiter="tif")
    logging.debug(file_names)

    logging.info("{} Start intensity analysis {}".format(50 * "=", 50 * "="))
    max_mean_intensity = 0
    max_intensity_frame = None

    for file_name in file_names:
        image_path = os.path.join(folderName, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mean_intensity = np.mean(image)

        if mean_intensity > max_mean_intensity:
            max_mean_intensity = mean_intensity
            max_intensity_frame = file_name

    print("Frame with maximum mean intensity:", max_intensity_frame)
    print("Maximum mean intensity:", max_mean_intensity)

    return max_intensity_frame, max_mean_intensity

def intensity_change(folderName: str, max_intensity_frame, max_mean_intensity):
    destFolder = os.path.join(folderName, "intensities")
    tools.makeFolder(destFolder)
    logging.debug(folderName)
    file_names = tools.fileLists(folderName, delimiter="tif")
    logging.debug(file_names)
    
    image_path = os.path.join(folderName, file_names[0])
    first_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print(image_path)
    image_path = os.path.join(folderName, max_intensity_frame)
    curr_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print(image_path)
    abs_diff_image = cv2.absdiff(first_frame, curr_frame)
    max_diff = np.max(abs_diff_image)
    max_diff_pixel = np.unravel_index(np.argmax(abs_diff_image), abs_diff_image.shape)
    print("Pixel with greatest change:", max_diff_pixel)
    print("Greatest change value:", max_diff)

    return max_diff_pixel

def generate_square_mask(folderName: str, pixel, max_intensity_frame):
    image_path = os.path.join(folderName, max_intensity_frame)
    curr_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    square_mask = np.zeros(curr_frame.shape)
    for i in range(pixel[0]-2,pixel[0]+2):
        for j in range(pixel[1]-2,pixel[1]+2):
            square_mask[i,j] = 1

    # mask_image = curr_frame*square_mask
    # average_intensity = np.sum(square_mask)/25

    # stats = {
    #     'Intensity_Within_Pixel': np.zeros(mask_image)
        
    # }

    # statsdf = pd.DataFrame(stats)

    return square_mask

# def cell_search(folderName: str, diameter: int):
#     # global rois
#     destFolder = os.path.join(folderName, "output")
#     tools.makeFolder(destFolder)
#     destFolder = os.path.join(destFolder, "cell_search")
#     tools.makeFolder(destFolder)
#     logging.debug(folderName)
#     fileNames = tools.fileLists(folderName, delimiter="tif")
#     logging.debug(fileNames)

#     logging.info("{} Start image analysis {}".format(50 * "=", 50 * "="))
#     logging.info("{} detect the brightest frame {}".format(50 * "-", 44 * "-"))
#     cyto_model = models.Cellpose(gpu = True, model_type="cyto")
#     for idx, fileName in enumerate(fileNames[:]):
#         if idx%100 == 0:
#             temp = cv2.imread(os.path.join(folderName, fileName), -1)
#             img = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#             masks = tools.detect_cells(img, diameter=120, model=cyto_model)
#             num_cells = masks.max()
#             cellInfo = tools.cell_info(masks, temp)
#             logging.info("idx->{},filename->{},numberOfCells->{}".format(idx, fileName, num_cells))
#             np.savetxt(os.path.join(destFolder, "{}.csv".format(fileName[:-4])), cellInfo, delimiter=",",
#                     header="index,x,y,signal,background,xmin,ymin,xmax,ymax", comments="")
#     logging.info("{} Finish image analysis {}".format(49 * "=", 49 * "="))

# def maximum_cells(folderName: str):
#     folderName = os.path.join(folderName, "output")
#     folderName = os.path.join(folderName, "cell_search")
#     fileNames = tools.fileLists(folderName, delimiter="csv")
#     df = pd.DataFrame()
#     for idx, fileName in enumerate(fileNames[:]):
#         if idx % 50 == 0:
#             logging.info("filename is {}".format(fileName[:-4]))
#         temp = pd.read_csv(os.path.join(folderName, fileName))
#         temp["file"] = fileName[:-4]
#         df = pd.concat([df, temp])

#     columns = df.columns
#     fileSns = df["file"].unique()
#     cellCounts = np.zeros(len(fileSns), )
#     for idx, fileSn in enumerate(fileSns):
#         cellCounts[idx] = len(df[df["file"] == fileSn])
#     maxIdx = np.argmax(cellCounts)
#     with open(os.path.join(folderName[:-12], "max_cell_frame.csv"), "w") as fp:
#         fp.write("max_cell_frame\n")
#         fp.write(str(maxIdx)+"\n")
#     logging.info("the maximum cell happened at frame {}".format(maxIdx))

#     plt.rcParams["font.size"] = "14"
#     style = "seaborn-v0_8-darkgrid"
#     plt.style.use(style)
#     plt.figure(figsize=(8, 6))
#     fig, ax = plt.subplots()
#     ax.plot(cellCounts, "-")
#     ax.plot(maxIdx, cellCounts[maxIdx], "-h", markersize=8)
#     ax.axvline(maxIdx, color="r", linestyle=":", lw=1.5)
#     ax.text(maxIdx + 5, cellCounts[maxIdx], str(maxIdx), color="r")
#     ax.set_xlabel("Frame")
#     ax.set_ylabel("Cell Count")
#     plt.savefig(os.path.join(folderName, "cell_count.png"), dpi=150)
#     plt.close()

def data_analysis(folderName: str, square_mask):
    logging.debug(folderName)

    destFolder = os.path.join(folderName, "output")
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
    prev_time = datetime(year=today.year, month=today.month, day=today.day, hour=prev_hr, minute=prev_min, second=prev_sec)
    for file in fileNames: 
        curr_hr = 0
        curr_min = 0
        curr_sec = 0
        curr_hr, curr_min, curr_sec = tools.time_stamp(file)
        curr_time = datetime(year=today.year, month=today.month, day=today.day, hour=curr_hr, minute=curr_min, second=curr_sec)
        timediff = curr_time - prev_time
        time_stamps.append(timediff.total_seconds())
        prev_hr = curr_hr
        prev_min = curr_min
        prev_sec = curr_sec
    # logging.info(time_stamps)
    logging.debug(fileNames)

    logging.info("{} Start image analysis {}".format(50 * "=", 50 * "="))
    logging.info("{} detect the brightest frame {}".format(50 * "-", 44 * "-"))

    
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

        filterImg = temp * square_mask
        # plt.imshow(temp, cmap='gray')
        # plt.show()

        df["roi"] = [np.sum(filterImg)/np.sum(square_mask)]
        result = pd.concat([result, df])

    result.to_csv(os.path.join(destFolder, "final_brightness.csv"), index=False)
    logging.debug("\n{}".format(result))
    logging.info("{} Finish image analysis {}".format(49 * "=", 49 * "="))


def post_analysis(folderName: str, threshold: float, square_mask):
    destFolder = os.path.join(folderName, "roi_stats")
    tools.makeFolder(destFolder)
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
    column = 'roi'
    
    logging.info(column)
    dataIn = np.array(df[column])
    baseline = df[column] - df['background']
    dataIn = baseline / baseline[0]

    energy, riseTime, fallTime, maxLoc, left, right = tools.timing_energy(dataIn, threshold=threshold)
    before_intensities = dataIn[0:int(left)]
    average_intensities = np.mean(before_intensities)
    peakLocation, peakVal, fwhm, leftIdx, rightIdx = tools.full_width_half_maximum(dataIn)
    temp = pd.DataFrame()
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
    plt.text(leftIdx + 1100, peakVal * .55, "FWHM is {:.3f}".format(fwhm))
    plt.text(left + 10, threshold + .7, "Rise time is {:.3f}".format(riseTime))
    plt.text(len(dataIn) // 2, threshold + .1, "Fall time is {:.3f}".format(fallTime))
    plt.title("{}".format(column))
    plt.text(left // 2, threshold + .4, "Average Intensity Before Peak: {:.3f}".format(average_intensities))
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Signal [DN]")
    plt.tight_layout()
    plt.xlim([time_stamps[0]-20, time_stamps[len(time_stamps)-1]+20])
    plt.savefig(os.path.join(destFolder, "statistics_{}.png".format(column)), dpi=150)
    plt.close()

    resultDf.to_csv(os.path.join(destFolder, "statistics.csv"), index=False)
    logging.info("{} Finish image analysis {}".format(49 * "=", 49 * "="))
    plt.autoscale()
    plt.show()
    plt.close()

def main(argv):
    print("Currently Running: RGC Ablation", flush=True)
    folderName = FLAGS.folder
    diameter = FLAGS.diameter
    mode = FLAGS.mode
    square_mask = np.zeros((512, 512))
    if mode == 0:
        max_intensity_frame, max_mean_intensity = intensity_search(folderName)
        pixel = intensity_change(folderName, max_intensity_frame, max_mean_intensity)
        mask = generate_square_mask(folderName, pixel, max_intensity_frame)
        image_path = os.path.join(folderName, max_intensity_frame)
        curr_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(curr_frame, cmap='gray')
        plt.imshow(mask, alpha = 0.5, cmap = 'gray')
        # plt.show()
        destFolder = os.path.join(folderName, "output")
        tools.makeFolder(destFolder)
        plt.savefig(os.path.join(destFolder, "mask.png"))
        # plt.close()
    # elif mode <= 1:
    #     cell_search(folderName, diameter)
    # elif mode <= 2:
    #     maximum_cells(folderName)
    if mode <= 1:
        data_analysis(folderName, mask)
    if mode <= 2:
        post_analysis(folderName, 1.1, mask)
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
