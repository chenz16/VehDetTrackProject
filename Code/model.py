'''
SVM model:
1) read labeled training data
2) extract training data features
3) normalize data
4) train SVM model
5) save model parameters.
'''



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



# Read in cars and notcars
# get data directory
veh_data     = cfg.Source['train_data_vehicles']
non_veh_data = cfg.Source['train_data_non_vehicles']

# read image
images_veh     = glob2.glob(veh_data + '*/*.png')
images_non_veh = glob2.glob(non_veh_data + '*/*.png')

cars = []
notcars = []
for image in images_veh:
    cars.append(image)
for image in images_non_veh:
    notcars.append(image)

# define feature extraction parameter
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 18  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

t=time.time()
car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
print('Number of training data', X_train.shape[0])
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

import pickle
dist_pickle={};
dist_pickle["svc"] =svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"]=orient
dist_pickle["pix_per_cell"]=pix_per_cell
dist_pickle["cell_per_block"]=cell_per_block
dist_pickle["spatial_size"] =spatial_size
dist_pickle["hist_bins"] =hist_bins
dist_pickle["color_space"] = color_space
dist_pickle["hog_channel"] = hog_channel
dist_pickle["spatial_feat"] = spatial_feat
dist_pickle["hist_feat"] = hist_feat
dist_pickle["hog_feat"] = hog_feat

pickle.dump(dist_pickle, open(cfg.Target['models'] + "svc_pickle.p", "wb" ) )
