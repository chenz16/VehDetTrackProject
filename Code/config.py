'''
this module is for configuration
this module defines:
image sources for train data ()
image sources for test images
target folder to store all processed image
target folder to store models
'''


Source= {}
Source['train_data_vehicles'] = '../train_data/vehicles/' # undistort sample image in this folder to show how camera cal works
Source['train_data_non_vehicles'] = '../train_data/non-vehicles/' # undistort sample image in this folder to show how camera cal works
Source['test_imgages'] = '../test_images/'

Target = {}
Target['output_images'] = '../output_images/' # place to store all processed images
Target['models'] = '../'
