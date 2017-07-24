---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./fig1.jpg
[image2]: ./fig2.jpg
[image3]: ./fig3.jpg
[image4]: ./fig5.jpg
[image5]: ./fig6.jpg
[image6]: ./fig7.jpg
[image7]: ./fig8.jpg
[image8]: ./fig9.jpg
[video1]: ./project_video_out.mp4
[video2]: ./test_video1_out.mp4
[video3]: ./test_video_out_2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! The code and plots can be seen in ./vehicle-detection.ipynb.
### Histogram of Oriented Gradients (HOG)
#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I started by reading in all the `vehicle` and `non-vehicle` images.  The code is there in "Load data cells".Here are some examples from each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. 
Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The figure shows the comparison of hog features between a car image and a non-car one. The code is there in `get_hog_features()` :

![alt text][image2]
Finally to compute the features of all the samples, I use the `extract_features()` in the **Extract Hog features** cell. It accepts a list of image paths and HOG parameters and produces a flattened array of HOG features for each image in the list.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of colorspaces X orientations X Pixels per cell X Cells per block X HOG channels and settled on the final choice based on the performance of SVC classifier produced using them.  
The final parameters I used are documented below:
- colorspace = 'YUV'
- orient = 11
- pix_per_cell = 16
- cell_per_block = 2
- hog_channel = 'ALL'

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have used only HOG features and didn't use either spatial intensity or channel intensity features. In the section titled "Train Classifier" I trained a linear SVM with the default classifier parameters. The linear SVM Classifier was able to achieve a test accuracy of 98.51%.

###Sliding Window Search

#### 1.Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the method `find_cars()` from the lecture content. The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) just once. Then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

To decide scale, window start and stop positions, I experimented with various combinations of parameters. I found out that small scales ~ 0.5 returned too many positives. So to reduce them, I used scales of 1,1.5,2 and 3.5. You can see them in the `Combine Sliding window search` cell of the Notebook. They contain the values. Below, You can see the rectangles returned by `find_cars()`
on one of the test images.

![alt text][image3]
Now, we know that a true positive is typically accompanied by several positive detections, while false positives are typically accompanied by only one or two detections. We employ a combined heatmap and threshold  to differentiate the two. The add_heat function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. To illustrate the above, first see the image heatmap with a false positive:

![alt text][image4]
Now, after applying threshold, we remove the false positve. You can see the thresholded heatmap as :

![alt text][image5]. 
The scipy.ndimage.measurements.label() function collects spatially contiguous areas of the heatmap and assigns each a label:

![alt text][image6].
Now finally, we take the minimum and maximum values of x and y co-ordinates to compute the final windows inside which more or less our cars lie. We get a result as: 

![alt text][image7] 

#### 2.Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The final implementation performs reasonably well with no false positives in the test images. Initially I chose only The Y channel of YUV image. It's accuracy never topped 97%, so I used all three channels of YUV image. That improved the accuracy to 98.51%. Also, increasing the pixels_per_cell from 8 to 16 greatly reduced the execution time. Other optimization techniques included changes to window sizing and overlap as described above, and lowering the heatmap threshold to improve accuracy of the detection.
The performance on `test images` are shown below: ![alt text][image8]

---
### Video Implementation

#### 1.Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4).You can see it here too: 



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for processing frames of video is contained in the cell titled "Pipeline for Processing Video Frames" and is identical to the code for processing a single image described above, with the exception of storing the detections (returned by find_cars) from the previous 15 frames of video using the prev_rects parameter from a class called Vehicle_Detect. Rather than performing the heatmap/threshold/label steps for the current frame's detections, the detections for the past 15 frames are combined and added to the heatmap and the threshold for the heatmap is set to 1 + len(det.prev_rects)//2 (one more than half the number of rectangle sets contained in the history) - this value was found to perform best empirically (rather than using a single scalar, or the full number of rectangle sets in the history).:
### Here is the output when each frame is processed separately:

[Test Video Frames processed separately][video2]

### Here is the output when we process information using the past frames:

[Test Video Frames processed using past rectangles][video3]

### Here is the final result:

[Final Project Video][video1]

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It was a good experience to learn about the sliding window method for object detection. However, I noticed that there are 2 disadvantages:
* The most important one being that it is computationally very expensive. It requires so many classifier tries per image. 
* Again for computational reduction not whole area of input image is scanned. So when road has another placement in the image like in strong curved turns or camera movements, sliding windows may fail to detect cars.

The [Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) might be a good alternative for the sliding windows approach.
