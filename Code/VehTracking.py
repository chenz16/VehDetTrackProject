
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

indx=0
for file in images:

    image = mpimg.imread(file)
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)

    image = image.astype(np.float32)/255
    scale = 1
    y_start_stop = (360, 656)
    window_size_MinMax = (40, 150)
    window_size_delt =8
    windows_size, ystarts, ystops = windows_yrestriction(window_size_MinMax, y_start_stop, window_size_delt)

    windows = []
    for window_size, ystart, ystop in zip(windows_size, ystarts, ystops):
        #y_start_stop = (ystart, ystop)
        windows_onesize = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=window_size, xy_overlap=(0.5, 0.5))

        windows = windows + windows_onesize

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

    out_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)
    heatmap = np.zeros((draw_image.shape[0], draw_image.shape[1]))
    heatmap = add_heat(heatmap, hot_windows)
    heatmap = apply_threshold(heatmap, 5)
    labels = label(heatmap)

    image = draw_labeled_bboxes(draw_image, labels)

    f0,(ax1, ax2) = plt.subplots(2,1, figsize=(20,10))
    ax1.imshow(out_img)
    ax1.set_title('Raw Detection')
    ax2.imshow(heatmap, cmap = 'gray')
    ax2.set_title('Image Heatmap')
    f0.savefig('../output_images/' + str(indx) + 'heatmap.jpg')


    f,ax = plt.subplots(1,1, figsize=(20,10))
    ax.imshow(image)
    f.savefig('../output_images/' + str(indx) + '.jpg')


    f3,ax = plt.subplots(1,1, figsize=(20,10))
    ax.imshow(out_img)
    ax.set_title('Results from SVM prediction')
    f3.savefig('../output_images/' + str(indx) + 'SVM.jpg')


    indx +=1
