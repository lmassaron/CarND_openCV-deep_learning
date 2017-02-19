READ ME
-------
The entire work on this exercise has been done achieving four stages:

1. Camera calibration
2. Unwarping corners
3. Converting to a thresholded binary image
4. Apply window fitler to remove artifacts in the image

The four steps have then be fused into a pipeline able to work on single images. The video processing has been added after providing the pipeline with a memory in order to stabilize the results, fasten processing and remove outliers.

CAMERA CALIBRATION
------------------
The calibration function is calculated using the calibration chessboard images provided in the repository. If information cannot be extracted from an image, then it is ignored. The returned function is indipendent and can be used later to undistort images in a pipeline because it incorporates the correct camera matrix and distortion coeઐcients.
A test is done to verify the undistort some of the test images (mostly the ones we couldn't use for calibration) and prove the procedure is correct. Differencing the undistorted and the original picuters visualizes the changes and provides an idea about how the image has changed.

UNWARPING CORNERS
-----------------
In order to manage the perspective transformation into a birds-eye view, an image of a straight lane road has been choosen. 
Four points have been manually located by try and error and then translated into a birds-eye perspective.
Naturally, from now on, we will work on undistorted images. Therefore distortion correction that has been calculated via camera calibration will be applied to each image.
The coefficients for unwarping corners  have been found out by various tries. The idea is that, after finding a correct transformation, if the camera has not been changed or moved from its initial position in the car, the transformation will hold for all other images. This is a strong assumption and one of the limitations of this exercise.
Since the procedure seems working, being the perspective rectified and having produced parallel lines, a function is produced for translating any image into bird-eye's view. This is possible because the computed camera matrix and coefficients have been incorporated into the function (basically it is a procedure with OpenCV functions very similar to camera undistorsion).
Moreover, since we will be needing it at a later stage, also the revere procedure is created, allowing passing back from bird-eye's view to the initial perspective.

THRESHOLDED BINARY IMAGE
------------------------
The next step is to create a binary image containing the two lanes, limiting the presence of extraneous artifacts. In order to achieve this result, two different approaches will be used, one based on HSV filtering colors (yellow and white) which compose the lanes, the other based on HLS Sobel filter (different directions: x and y on different channels, L and S, the two most resistant to different light coditions).
The yellow and white masks have been derived from a public post of Vivek Yadav: https://medium.com/@vivek.yadav/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3#.n08e8azfi
Combining correctly the sobel filters took some time, in the end S chanel vertical and horizontal and L chanel horizontal seemed to work the best.
As before, the resulting binary images representing the lanes are checked against the test pictures available. The match is verified by visually comparing the original picture and the transformed one.

APPLY WINDOW FILTER TO REMOVE ARTIFACTS IN THE IMAGE
----------------------------------------------------
In order to remove any spurious artifacts, distinguish each lane and model them into functions by a polynomial fitting, these steps are taken:

* the image is sliced into image windows with height no more than 40 pixels
* each slice, turned into an histogram is checked for peaks, revealing the presence of the two lanes this is achieved by OpenCV function find_peaks_cwt
* when a peak is found, a function finds out the extend of pixels around it (a peak is treated like a centroid)
* being near to the right or left border helps locating if it is the left or right lane
* the pixel extent around the peak is used to carved out a box in the slice where high is the expectation to find a lane
* inside the carved out box a certain amount of pixels are sampled
* when the procedure has finished passing all the sliced image windows, it should have two series of points corresponding to the left and right lane
* since there is still possibility that some strange artifact is present in the series, a filtering is applied: the points are ordered and check one after the other, if any of them is too different from the previous one, it is excluded
* the filtering procedure also figures out the length of sampled points helping to understand if it a continuous lane or a discontinued one
* after filtering the two series a polynomial fitting (a quadratic linear regression) is done for both resulting in two coefficients vectors and in two functions
* by doing so, both curvature and distance to the center of the lane are calcualted (accordingly to the formulas provided in the lessons). The radius of curvature is be given in meters, assuming the curve of the road follows a circle and the position of the vehicle within the lane is given as centimeters off of center.
* the functions are used to project the lanes on the bird-eye view and to mark the lane area, the projection is then rendered prospective again and fused with the original image

COMBINING INTO A PIPELINE
-------------------------
The previous pipeline, suitable for single images, is rendered for video streams. A class, Memory, helps keeping track of all the infomation on lanes previously found. No previous information is hard-coded. The idea is to use moving averages to stabilize measurements on curvature and deviation from center and to check regression coefficients for stability in respect of previous solutions (in case previous solutions are used or averaged with the actual one).

This is possible after a few frames of the video, when the lanes are recognized with confidence and the information about their location and shape is transmitted and propagated by means of the regression coefficients representing their shape.

Keeping track of past of lanes by their previous regression coefficients allows for correction of outliers, smoothing results and saving time on the detection because the window filter is applied only once every three frames (less frequent usage of the filter led to some distorsions).  

CONCLUDING REFLECTIONS
----------------------
As a conclusion, I cannot but reflect on how the pipeline works fine with the exercise's video but failed on the two more challenging videos. There a few things that can be improved therefore:

* resistance to different road conditions and kind of lane marks (double mark, too think white mark)
* too strong curvature, lanes disappearing because of curving
* artifacts on the lane
* deep shadows

Possible improvements for the pipeline that could immediately address the above mentioned problems are:

* exploring other color spaces, even less sensible to dark or light conditions. A good candidate could be the normalized RGB https://en.wikipedia.org/wiki/Rg_chromaticity
* erroneous lane detections are filtered by smoothing the results and taking into account previous frames. I would now introduce a filtering at image level in order to figure out if the estimate curvature is plausible or not and derive the unplausible estimations of a lane boundary using the other one. That would fix the lane detection problems when a lane disappears or when for some reason the detected lanes are distorted.

I repute that the core problem with this exercise is that all the pipeline is build from images from the same, single video. More videos should be tested in order to assure the robustness of the pipeline.
