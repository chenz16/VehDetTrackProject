# VehDetTrackProject

## 1. Submission packages
### Code folder:

vehTracking.py: script to process sample images for vehicle detection and tracking

pipe_line.py -  pipe line to process video for vehicle detection and tracking

process.py -  Defines the core functions to process images for vehicle detection and tracking

config.py- define basic input and output source/address of image process. 

writeup_report/README - Explain what is included for the submission and how it is done. 

### output_images folder


### project_video_VehTrack.mp4

project submission video which overlays detected vehicle bounding box to original images/video

## 2. Go through rubrics


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function "get_hog_features `some_file.py`

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Alt text](/output_images/TrainingSample.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=18`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![Alt text](/output_images/HOG_Features.jpg)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried different color space options and there is a little difference between each other. Based on several tries, here are my final choosing:

    color_space = 'YCrCb' 
  
    orient = 18  # HOG orientations
  
    pix_per_cell = 8 # HOG pixels per cell
  
    cell_per_block = 2 # HOG cells per block
  
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
  
Using "YCrCb" as color space can easily detect the right side white car.  I tried differnt orient number. I finally choose 18 which gives decent result. Increasing the number of orient significantly increases the features numbers and therefore, the computing time for vehicle detection and tracking. 

I did not use spatial (color) features as i found they are not very helpful to the problem. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color histogram and HOG features. I also tried different combinations. In most of the cases, the prediction accuacy in test data set could reach 98-99%. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

There are several kety steps to implement a sliding window approach:

   ####1) in general the search space is only limited to (rougly speaking) bottom half of the image space. This is defined by: y_start_stop = (360, 656)
   
   ####2) the searching space is further limited to a smaller one depending on the size of the windows. In general, we only want to apply a large window for the near bottom space and apply small window near the middle y space. This method is applied through "windows_yrestriction" in process.py
   
   ####3) at a given window size and searching space in a image, get the windows left up corner and bottom right corner point. This function is performed by:
   
   def windows_yrestriction(window_size_MinMax, y_start_stop, window_size_delt):

    windows_size = []
    window_xsize_min = window_size_MinMax[0]
    window_xsize_max = window_size_MinMax[1]
    window_ysize_max = window_xsize_max
    window_ysize_min = window_xsize_min

    window_xsize_delt = window_size_delt
    window_ysize_delt = window_size_delt

    window_xlist = np.arange(window_xsize_min, window_xsize_max,window_xsize_delt)
    window_ylist = np.arange(window_ysize_min, window_ysize_max,window_ysize_delt)

    b2 = y_start_stop[0]
    b1 = y_start_stop[1] - 1.5*window_ysize_max
    a2 = window_xsize_min
    a1 = window_xsize_max
    ystarts = (b1-b2)/(a1 - a2) *(window_ylist - a2) + b2
    ystarts = ystarts.astype(np.int)

    b2 = y_start_stop[0] +1.5*window_xsize_min
    b1 = y_start_stop[1]
    a2 = window_xsize_min
    a1 = window_xsize_max
    ystops = (b1-b2)/(a1 - a2) *(window_ylist - a2) + b2
    #print(window_ylist)
    #print('ystarts', ystarts, 'ystops', ystops)

    for window_xsize, window_ysize in zip(window_xlist, window_ylist):
        windows_size.append((window_xsize, window_ysize))

    return windows_size, ystarts, ystops

An visualization of search windows is as the follows:

![alt text](/output_images/Sliding_window.jpg)

Note: I did not use the method of HOG subsampling, which was introducted in the course material. HOG subsmapling should reduce the computing time of extracting HOG feature for the vehicle detection. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text](/output_images/0SVM.jpg)
![alt text](/output_images/1SVM.jpg)
![alt text](/output_images/2SVM.jpg)
![alt text](/output_images/3SVM.jpg)
![alt text](/output_images/4SVM.jpg)
![alt text](/output_images/5SVM.jpg)

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](/test_video_VehTrack.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I also created an object 'LastFrame' from the class 'FrameInfo' which records the heatmap information from the last step.  From the positive detections I created a heatmap, which is fused with last step heatmap. I then applied a threshold to heatmap.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here is the code sample:

    heatmap = np.zeros((draw_image.shape[0], draw_image.shape[1]))
    heatmap = add_heat(heatmap, hot_windows)
    if LastFrame.First is False:
        heatmap = 0.5*heatmap + 0.6*LastFrame.heatmap
    else:
        pass

    heatmap = apply_threshold(heatmap, 2.3)
    labels = label(heatmap)
    image = draw_labeled_bboxes(draw_image, labels)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:




### Here are six frames and their corresponding heatmaps:

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text](/output_images/0heatmap.jpg)
![alt text](/output_images/1heatmap.jpg)
![alt text](/output_images/2heatmap.jpg)
![alt text](/output_images/3heatmap.jpg)
![alt text](/output_images/4heatmap.jpg)
![alt text](/output_images/5heatmap.jpg)

### Here the resulting bounding boxes are drawn :
![alt text](/output_images/0.jpg)
![alt text](/output_images/1.jpg)
![alt text](/output_images/2.jpg)
![alt text](/output_images/3.jpg)
![alt text](/output_images/4.jpg)
![alt text](/output_images/5.jpg)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When proper color space is selected (here YCrCb), the SVM is able to capture the vehicle through sliding window technique. However, the original SVM detection has a lots of false positive detection. 

In order to reduce the number of false positive and make the detection more robust, additional voting mechanisim are employed:

        1) apply proper threshold value to elimate those detections in which the SVM in sliding windows may ocationaly detects something but not the vehicle 
        2) combine the current step detection and the one of last step. The assumption here is vehicle in a frame would not move too much in next frame (considering fps her = 25). The heatmap in last frame could be use as voters of current frame detection. Again, proper threshold value should be applied 
        3) when we draw the final bounding box, very small box is removed to elimate the fasle positive detection. The mechanism of false positive small box is due to vote mechanism we used. Sometime, voting mechanism yield small area when the detection is positve. Apparanetly, these small areas not not trusted as vehicle detection .

