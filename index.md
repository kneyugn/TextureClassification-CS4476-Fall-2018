## Classifying Textures: Proposal

### Team: Daham Eom, Abigail James, Kim Nguyen, Zohra Tabassum
Link to project site: [https://kneyugn.github.io/TextureClassification-CS4476-Fall-2018/](https://kneyugn.github.io/TextureClassification-CS4476-Fall-2018/)

Link to project repository: [https://github.com/kneyugn/TextureClassification-CS4476-Fall-2018](https://github.com/kneyugn/TextureClassification-CS4476-Fall-2018)

### Abstract
This project aims to compare four different feature extraction algorithms that include Local Binary Pattern (LBP), Scale-Invariant Feature Transform (SIFT), Speeded Up Robust Features(SURF), and Gray Level Co-Occurrence Matrix (GLCM). Then, bag-of-features model is applied with SVM to classify the images. At the midpoint of this project, all algorithms have been implemented.

### Introduction
The purpose of the project is to identify 28 classes of texture from the Kylberg Texture Dataset [11]. The overarching goal is to explore feature detection algorithms in the domain of texture classification. The input of the system will be an image of the texture type. Then, the system will produce vector features from the following algorithms: Local Binary Pattern, Scale-Invariant Feature Transform, Speeded Up Robust Features(SURF), and Gray Level Co-Occurrence Matrix. Then, these feature vectors are passed into machine learning algorithms that is SVM to classify the type of texture. The output of the entire system is the prediction of the type of texture. In this project, we aim to implement the feature extraction algorithms. Then, we will use the bag-of-words model to classify the images using SVM. We will then compare and analyze each feature algorithm’s performance based on precision, recall and accuracy scores.

### Approach
The first half of our project is focused primarily on implementing the algorithm from pseudocode, resources and research papers to understand the algorithms. We will produce some classification results from Support Vector Machine, but it is not yet the focus at this stage. For the second half, we plan to fully implement and standardize the bag-of-features approach with SVM. Thus, the following are the design choices that we have made to implement the algorithms and the approaches we have taken for the midpoint results.

#### The datasets:
From The Kylberg Texture Dataset [8], we are producing preliminary results from a small collection of the dataset. Instead of looking at all 28 classes to test our implementation of computer vision algorithms, we are training on only on 6 texture classes for the midpoint update and plan to expand to all 28 classes in the final report. There are 40 pictures in each class. Overall, 240 pictures were used. These 6 classes include canvas, cushion, linseeds, sand, seat, and stone. While the image sizes are originally 576x576, we have resized the images to 100x100 to begin. This is due to the fact that SIFT features produce a vector of size 128 for each feature found. A picture of size 100x100 can produce over 100 features. Additionally, each picture is producing 24 images of varying octaves and scales. For LBP, what is produced as the feature is an arrays of histograms for the different images from the dataset. The histogram has a bin range from 0-255 representing the different unique LBP binary values counted from calculation, and the frequency of these values. Thus, we find that 100x100 is a good starting size to begin using to test and refine our algorithms. 
We then split this training set with 60/40, 70/30, and 80/20 as fitting/predicting data sets. From the fitting set, we feed each image into one of the four feature detection algorithms (Local Binary Pattern, Scale-Invariant Feature Transform, Law’s Texture Energy Measures, and Gray Level Co-Occurrence Matrix). Then, we pass the array of features from each individual algorithm into the Support Vector Machine to fit the data. Then, we use the predicting data set and predicted the classes from the given data. We then evaluated the results by accuracy, precision and recall.
We use Support Vector Machine because it is often used for classification with feature vectors, and there are prior research examples of SVM being used with the algorithms that are in our proposal [4][14][13][2][7].

#### The algorithms:

##### SIFT:
SIFT:  We looked at 4 octaves of quartered, halved, normal and double sizes which contributes to the image scale invariant aspect of the algorithm. There are 6 scales or 6 gaussian blurring intensities applied at each octave. As described in David Lowe’s work, each increasing scale is increased by multiplicative factor of k [1][5][15]. The value for k is chosen as 1.41424 as a starting point. For each increasing octave with the same scale level, the sigma is also a multiplicate of k. This process is to ensure that we detect dominant features that are dominant across all scales [1]. Then, the difference-of-gaussian function is used to provide true scale invariance [1]. This produces 5 difference of gaussian scale per octave. Then, local extrema detection is performed. Per octave, for each point in 4 difference of gaussian scale level, 9 neighbors from the previous scale, 9 neighbors from the next scale, and 8 neighbors from the current scale are examined or a total of 26 neighboring scales that are within 1 pixel from the point. If the current pixel is the minimum or maximum of all neighbors, it is chosen as a minimum or maximum point. Additionally, each point must pass a threshold to be examined as a an  minimum or maximum point. Then, for each image across all octaves and scales, orientation and magnitude are found. For each of these extrema points, a keypoint is produced which contains information about the location of the keypoint, its gaussian sigma, and its dominant orientation. Then, descriptors are produced from each keypoint. For each keypoint, a descriptor/feature vector of size 128 is produced. It is essentially a window of size 16x16 pixels surrounding the feature pixel. Then, it is a description of 16 4x4 histograms in which each histogram contains 8 bins, and each represents an orientation of 45 degrees. While the majority of the implementation is cross checked between the following resources including Lowe’s paper and tutorials on SIFT, existing code implementation is used to guide us when critical [15][5]. This is especially important in finding the gaussian window weight to give more priority to orientation that are closest to the feature pixel. Further implementation will include clustering the SIFT features and using a histogram of SIFT features in a bag-of-features model to classify with SVM.

##### SURF
Speeded Up Robust Features (SURF) is the another algorithm that is being analyzed in this project. SURF is similar to SIFT but uses integral images and a Hessian Matrix to cut down on the calculation time. The SURF algorithm is based on two separate steps, the first being to find the interest points in the image and then the second is creating a descriptor which can be used for matching  features [10]. The first step in finding the interest points is to calculate the integral image, where each pixel is the sum of every pixel above and to the left[9]. Four octaves, with 4 scales were used to calculate a size parameter which was used to calculate the determinant of Hessian Matrix [9]. The next step is to use non-maximum suppression to remove any points that are not maximum in their neighborhood of eight pixels. Then Taylor Expansion is used to refine each point[9]. The next step is to find the orientation of the interest point in the neighborhood[9]. The last step is to return a Surf Descriptor as well as the sign of the laplacian and the orientation of the keypoint[9]. This information is what can be used to test the algorithm on different textures.
	Unfortunately, this algorithm is not fully functional for this midpoint check in. At first many of the sources online only gave a brief description of the algorithm, that did not give a enough details for implementation. However a source that contained pseudocode and a much more detailed description was found [9]. While all of the code has been written at this point, something is not working correctly when calculating the interest points. The fall image shown below [11] is the original image that was passed in to test the code. The photo on the left is the grayscale version and the red circles represent the interest points. The interest points are only showing up on the edge of the image, primarily in the lowered numbered columns. At first this error  was believed to be from miscalculations when creating the integral image, but after making adjustments there were still issues.Next the threshold was lowered to 10^2 instead of 10^3 which is recommended by the paper and while that did allow for more circles, it still was not working correctly [9]. Future areas to look into include going over all of the equations one more time to see if there were any errors in coding them. Another area potential source of error could be that the code does not account for the origin of the image being the top left and not the bottom left like it is in other programs. Though there was an effort to account for that when coding there are many places in the algorithm where equations have to be reworked to account for that, so there could be places where this was not done properly.. By the completed version of this project, these issues should be gone and SURF can be tested along with the other algorithms. 
  
  ![Image](https://i.imgur.com/OyQ6OHi.png)

##### Local Binary Pattern
You want to calculate the lbp value for each pixel. Iterating through the grayscale image version of the pic, you set the current pixel as the center pixel for your 3x3 neighborhood. You threshold the the neighborhood by comparing the center pixel values to all the neighbors, and if the center less than the neighbor, set the threshold to one. Else set the value to zero. The LBP value you get is the 8-binary neighborhood of the center pixel converted to a decimal representation. To get this representation, start in a clockwise fashion and the top right neighbor as 2^0 and continue from there. Multiple by the threshold values. That is your LBP value. 
 ![Image](https://i.imgur.com/VDVbcBe.png)

##### Gray Level Co-Occurrence Matrix
In this midterm update, a dataset of 6 textures with 40 samples in each class is used to obtain GLCM values and corresponding statistical metrics. The GLCM is a n x n matrix where n is the number of gray level in an image and each cell represents the co-occurrence of gray-level pixel in an image. Thus, all image samples are converted to 8 bit grayscale images and a GLCM is calculated for each converted image. Multiple GLCM calculations are done with displacement vector of [1, 2, 4] and rotation vector of [0] to find the optimal combination for statistics calculations. 3 possible sets of GLCMs are normalized by the number of pixels and then used to calculate the values of contrast, dissimilarity, homogeneity, ASM, energy, and correlation of the texture image by following equations. [7]

 ![Image](https://i.imgur.com/Pfo51Fc.png)

Source for Figure: [16]
You then get the 2-Dimensional Image Representation of the LBP. You can use this to calculate a histogram for that image.  Next is constructing the histogram of the frequencies of the LBP values calculated (in percentage). Since the 2-D Since a 3 x 3 neighborhood has 2 ^ 8 = 256 possible patterns, the LBP 2D array thus has a minimum value of 0 and a maximum value of 255,  and so a 256-bin histogram is constructed of LBP codes [3]. Each image will return a histogram, which is just a one dimensional array. You use this as the feature of the images, along with the label set of images from  1-6 for each image that represents the different folders of textures. I then used SVC, and then used a test set of the histograms and labels, to see the predicted labels that the model would get. And then compared the original labels and the predicted labels. 

After getting all values, a training set and test set for SVM model are constructed with above statistics as attributes and 6 textures: canvas, cushion, linseed, sand, seat, and stone as labels. The dataset is divided into a training set and a test set with ratio of 8:2, 7:3, and 6:4 and feeded to three different SVM implemented in Python - sklearn libraries. The result models are sensitive to input images, hyperparameters, and SVM algorithms. To maximize the overall accuracy of the model, three types of SVM, SVC, linearSVC, and NuSVC, are run with different parameters settings such as penalty value, kernel type, and gamma value. Also, the original images and resized images are tested to check the influence of different image input.

### Experiments and results

#### SIFT

 ![Image](https://i.imgur.com/4vqPfgd.png)
 
 ![Image](https://i.imgur.com/TvDEnM5.png) 
 
 ![Image](https://i.imgur.com/Po6KDPG.png)
 
 ![Image](https://i.imgur.com/XNHTPxJ.png)
 
Above are the images of features extracted of varying octaves and scales. There are 4 octaves and 6 scales. The rate that gaussian sigma is multiplied is k = 1.414214. The starting sigma is .7. For the first two octaves, no features were found throughout all the scales. Only a few features were found in the normal size octave and most were found in the doubled sized octave. In the next stage, we will look at image sizes that are larger than 100x100 pixels to ensure that all octaves produce features to ensure true image scale invariant. 

 ![Image](https://i.imgur.com/Sn149Et.png)

Here, the baseline of only passing in an array of descriptors into SVM to fit and predict. We have roughly 50% accuracy rate for all data splits. The precision and recall scores are very similar as well. This shows that on average, relevant classes are correctly identified at around 50% of the time (recall). On average for all classes, 50% of those identified for a class are correct (precision). 

The results are what we expected if a bag-of-feature model is not employed, thus these are the baseline scores. Instead of passing in the array of descriptors, we must first cluster the features with K-means. Then, map the label to cluster for descriptors in each image. Each image will have a histogram of how often these descriptors show in the image. Then, an array of these histogram features is used to fit and predict the classification for these images.

#### LBP

 ![Image](https://i.imgur.com/BgpiFCj.png)

Here are the corresponding histograms for these images:
![Image](https://i.imgur.com/PXxNy1A.png) 
![Image](https://i.imgur.com/aIz3zdz.png) 
![Image](https://i.imgur.com/2g5PI4S.png) 
![Image](https://i.imgur.com/vCzDcC4.png)


SVM Results for LBP:
![Image](https://i.imgur.com/Yl1C1q8.png)

The results show very high accuracy and precision, and we see the accuracy increase as the training set increases and then decrease. This is a key sign of overfitting, which is causing such high results and the accuracy to plateau and then decrease as the training set increases. Future work would be to see why this overfitting is occurring, and using cross validation to combat it. 

#### SURF
2D LBP representation of different image texture


### GLCM 

Statistics example from GLCM with 100 x 100 size image and d = 1

![Image](https://i.imgur.com/2czPsg5.png)

Statistics example from GLCM with 576 x 576 size image and d = 1
![Image](https://i.imgur.com/U8Hrdi0.png) 

It can be noticed that GLCM algorithm does not extract distinctive features when the resized images are used. Therefore, only images with original size are used to make a SVM classifier.
SVM Results for GLCM
Altering displacement vector in linear kernel type

![Image](https://i.imgur.com/JoxDsMw.png) 

Since the result of LinearSVC is too low, it is omitted and only SVC and NuSVC are used for further optimization.

![Image](https://i.imgur.com/juODoeB.png) 


GLCM with displacement factor of 1 results the highest accuracy. Thus, d = 2 and 4 are not tested for the rest of the experiments.
Result of using different kernel type

![Image](https://i.imgur.com/XVKQpvk.png)


NuSVC performs better than regular SVC so hyperparameter tuning is only done with the NuSVC and linear kernel.



Result of using different gamma values

![Image](https://i.imgur.com/jkmScwx.png)

Result of different data split

![Image](https://i.imgur.com/KVlwIkL.png) 

Final result with optimal settings

![Image](https://i.imgur.com/A9iiQ74.png)

This table shows the performance of SVM using features obtained from GLCM with displacement 1. The type of SVM is NuSVC with gamma = ⅙ and linear kernel. Even though the SVM and GLCM are tuned with various parameters, the final accuracy is only about 65% which can be considered as a low value. Moreover, the SVM model performs poorly with classifying Linseeds and stone textures and it cannot classify cushion texture. However, the SVM model shows high accuracy for classifying canvas, sand, and seat textures.


### Conclusion

We will extend to classify all 28 classes and increase the size of the images in the dataset. We will also extend how we process the algorithms. For SIFT, qualitative results show that the first two octaves do not produce any feature. This means that the implementation is not truly scale invariant. Thus, we will increase the image size until we can find meaningful features across all scales. We will also explore with tuning the k and sigma parameters. Finally, we will properly implement the bag-of-words model to improve accuracy, precision, and recall. For LBP the accuracy is extremely high, due to overfitting. Using cross validation and increasing the number of features used to accurately predict the label set might be several good ways to combat it, and further investigation into this is required. The SVM using features extracted from GLCM shows low overall accuracy. However, the model classifies three out of six textures with significantly high accuracy while it generally fails to classify other three textures. For the future implementation, using more image samples and additional statistics values can increase the overall accuracy. Also, more thorough hyperparameter tuning process is required with cross-validation to generate a better model. Overall, for all the algorithms, there needs to be further investigation using cross validation and  tuning of the parameters to combat the underfitting and overfitting that is occuring. For the algorithms that are underfitting, and having low accuracies, increasing the size to use the full dataset instead of partial will help increase the accuracy as well. And finally, as discussed in the results section for the Speeded Up Robust Features  algorithm, there is a bug that is preventing the interest points from being calculated correctly. The code needs to be looked at further to determine what exactly is causing this issue so it can be removed. Once that issue has been taken care of , SURF can be analyzed like the other algorithms. If this issue is not resolved, we will find another algorithm to analyze for texture classification. 

### References

[1] Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints.		 	 International Journal of Computer Vision,60(2), 91-110. 					 	https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

[2]Kim, D., & Dahyot, R. (2008). Face Components Detection Using SURF Descriptors and 		SVMs. 2008 International Machine Vision and Image Processing Conference.  			doi:10.1109/imvip.2008.15

[3] Local Binary Patterns with Python & OpenCV. (2018, June 21). Retrieved from 			 https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/

[4] Moranduzzo, T., & Melgani, F. (2012). A SIFT-SVM method for detecting cars in UAV 		images. 2012 IEEE International Geoscience and Remote Sensing Symposium.  		  	doi:10.1109/igarss.2012.6352585

[5] Sinha, U. (n.d.). Generating a feature. Retrieved from   							 http://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/

[6]Statistical Texture Measures Computed from Gray Level ... (n.d.). Retrieved from   		         ………http://www.docucu-archive.com/Statistical-Texture-Measures-Computed-from-Gray-Lev   	  el.pdf

[7] 1.4. Support Vector Machines. (n.d.). Retrieved from   							http://scikit-learn.org/stable/modules/svm.html

[8] G. Kylberg. The Kylberg Texture Dataset v. 1.0, Centre for Image Analysis, Swedish 			 University of Agricultural Sciences and Uppsala University, External report (Blue series) 	  No. 35. Available online at: http://www.cb.uu.se/~gustaf/texture/

[9]Oyallon, E., & Rabin, J. (2015). An Analysis of the SURF Method. Image Processing On 		 Line, 5, 176-218. Retrieved October 29, 2018, from  						 http://www.ipol.im/pub/art/2015/69/?utm_source=doi

[10]Birchfield, S. (n.d.). SURF detectors and descriptors [PPT]. Clemson: Clemson University.  	

[11] Fall [Digital image]. (n.d.). Retrieved October 29, 2018, from     			  ………https://www.almanac.com/content/2018-fall-foliage-forecast-vivid-northeast-mixed-bag-		   elsewhere 

[12] Module: Feature¶. (n.d.). Retrieved October 31, 2018, from 						 http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycomatrix

[13] Moranduzzo, T., & Melgani, F. (2012). Iris Recognition using Gray Level Co-occurrence 		 Matrix and Hausdorff Dimension. 2012 IEEE International Geoscience and Remote 		 Sensing Symposium. doi:10.1109/igarss.2012.6352585

[14] Qian, K., Zhang, Y., & Hasegawa-Johnson, M. (2016). Application of local binary patterns 		 for SVM-based stop consonant detection. Speech Prosody 2016. 					 doi:10.21437/speechprosody.2016-229

[15] R. (n.d.). Rmislam/PythonSIFT. Retrieved October 31, 2018, from    ……….https://github.com/rmislam/PythonSIFT/blob/master/siftdetector.py

[16] Katsigianni, S. (n.d.). [Example of Local Binary Pattern]. Retrieved from   	 ……....https://www.researchgate.net/figure/Example-of-Local-Binary-Pattern-calculation-for-a-3		-3-pixel-neighbourhood-a-Greyscale_fig2_260261605
