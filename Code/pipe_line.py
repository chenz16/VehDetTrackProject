
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import config as cfg
import glob2
from process import *
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load( open(cfg.Target['models'] + "svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
hog_channel = dist_pickle["hog_channel"]
spatial_feat =dist_pickle["spatial_feat"]
hist_feat =dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]


images = glob2.glob(cfg.Source['test_imgages'] + '*.jpg')

LastFrame = FrameInfo()


def VehTrack(image):
    draw_image = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)

    image = image.astype(np.float32)/255
    # define windows
    y_start_stop = (360, 656)
    window_size_MinMax = (40, 150)
    window_size_delt =8
    windows_size, ystarts, ystops = windows_yrestriction(window_size_MinMax, y_start_stop, window_size_delt)

    windows = []
    for window_size, ystart, ystop in zip(windows_size, ystarts, ystops):
        y_start_stop = (ystart, ystop)
        windows_onesize = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=window_size, xy_overlap=(0.5, 0.5))

        windows = windows + windows_onesize

    # search windows containing vehicles based on SVM
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    # draw windows in image
    out_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)

    # reselect window based on multiple detection
    heatmap = np.zeros((draw_image.shape[0], draw_image.shape[1]))
    heatmap = add_heat(heatmap, hot_windows)
    if LastFrame.First is False:
        heatmap = 0.5*heatmap + 0.6*LastFrame.heatmap
    else:
        pass

    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    image = draw_labeled_bboxes(draw_image, labels)
    #image = draw_labeled_bboxes(out_img, labels)

    LastFrame.First = False
    LastFrame.heatmap = heatmap
    return image


output = '../project_video_VehTrack.mp4'
clip = VideoFileClip('../project_video.mp4')

VehTrack = clip.fl_image(VehTrack)
VehTrack.write_videofile(output, audio=False)

'''
output = '../test_video_VehTrack.mp4'
clip = VideoFileClip('../test_video.mp4')

VehTrack = clip.fl_image(VehTrack)
VehTrack.write_videofile(output, audio=False)
'''
