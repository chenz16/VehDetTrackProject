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
import matplotlib.pyplot as plt
import pickle



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
# Read in cars and notcars
# get data directory
veh_data     = cfg.Source['train_data_vehicles']
non_veh_data = cfg.Source['train_data_non_vehicles']

# read image
images_veh     = glob2.glob(veh_data + '*/*.png')
images_non_veh = glob2.glob(non_veh_data + '*/*.png')


indx_veh = np.random.randint(1, len(images_veh ))
indx_non_veh = np.random.randint(1, len(images_non_veh))

image_veh = mpimg.imread(images_veh[indx_veh])
image_non_veh = mpimg.imread(images_non_veh[indx_non_veh])


f, (ax1, ax2) = plt.subplots(1,2, figsize = (20, 10))
ax1.imshow(image_veh)
ax1.set_title('vehicle training sample ')
ax2.imshow(image_non_veh)
ax2.set_title('non-vehicle training sample')
f.savefig(cfg.Target['output_images'] + 'TraingSample.jpg')



# define feature extraction parameter
'''
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
'''
y_start_stop = [None, None] # Min and max in y to search in slide_window()


'''
hog_features_3channel=np.array([])
f2, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (30, 10))

image_channel = image_veh[:,:,0]
hog_features = get_hog_features(image_channel, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True)
ax1.plot(hog_features)
ax1.set_title('Y Channel HOG Features')
image_channel = image_veh[:,:,1]
hog_features = get_hog_features(image_channel, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True)
ax2.plot(hog_features)
ax2.set_title('Cr Channel HOG Features')

image_channel = image_veh[:,:,2]
hog_features = get_hog_features(image_channel, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True)
ax3.plot(hog_features)
ax3.set_title('Cb Channel HOG Features')
f2.savefig(cfg.Target['output_images'] + 'HOG_Features.jpg')
'''

y_start_stop = (360, 656)
window_size_MinMax = (40, 150)
window_size_delt =8
windows_size, ystarts, ystops = windows_yrestriction(window_size_MinMax, y_start_stop, window_size_delt)

image_address = glob2.glob(cfg.Source['test_imgages'] + 'test1.jpg')
image = mpimg.imread(image_address[0])


windows = []
for window_size, ystart, ystop in zip(windows_size, ystarts, ystops):
    y_start_stop = (ystart, ystop)
    windows_onesize = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=window_size, xy_overlap=(0.5, 0.5))

    windows = windows + windows_onesize

for bbox in windows:
    cv2.rectangle(image, bbox[0], bbox[1], [150, 150, 150], 6)

f3, ax = plt.subplots(1,1, figsize = (30, 10))
ax.imshow(image)
f3.savefig(cfg.Target['output_images'] + 'Sliding_window.jpg')


#plt.show()

#hog_features_3channel  = np.hstack((hog_features_3channel, hog_features))

#print(hog_features_3channel.shape)
