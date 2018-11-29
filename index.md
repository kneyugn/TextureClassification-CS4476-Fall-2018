## Classifying Textures: Midpoint Update

![Image](https://i.imgur.com/FZhRIqF.png)

### Team: Daham Eom, Abigail James, Kim Nguyen, Zohra Tabassum
Link to project site: [https://kneyugn.github.io/TextureClassification-CS4476-Fall-2018/](https://kneyugn.github.io/TextureClassification-CS4476-Fall-2018/)

Link to project repository: [https://github.com/kneyugn/TextureClassification-CS4476-Fall-2018](https://github.com/kneyugn/TextureClassification-CS4476-Fall-2018)

### Abstract
This project aims to compare three different feature extraction algorithms that include Local Binary Pattern (LBP), Scale-Invariant Feature Transform (SIFT), and Gray Level Co-Occurrence Matrix (GLCOM). Then, bag-of-features model is applied with SVM to classify the images. The results show that LBP provides the most accurate results with  minimal running time to return results.

### Introduction
The purpose of the project is to identify 28 classes of texture from the Kylberg Texture Dataset [11]. The overarching goal is to compare feature detection algorithms in the domain of texture classification. Previous studies only focused on looking at how well one algorithm did in texture classification. However, we will be looking at several different computer vision algorithms for image identification, to see which produces the most efficient and best results for identifying the textures. The input of the system will be an image of the texture type. Then, the system will produce vector features from the following algorithms: Local Binary Pattern (LBP), Scale-Invariant Feature Transform (SIFT), Gray Level Co-Occurrence Matrix (GLCOM). Then, these feature vectors are passed into machine learning algorithms that is SVM to classify the type of texture. The output of the entire system is the prediction of the type of texture. In this project, we aim to implement the feature extraction algorithms. Then, we will use the bag-of-words model to classify the images using SVM. We will then compare and analyze each feature algorithm’s performance based on precision, recall and accuracy scores. While other studies have looked at these algorithms individually, we aim to look at one data-set with different algorithms to compare their results.

### Approach
The first half of our project was focused primarily on implementing the algorithms from pseudocode, resources and research papers to understand the algorithms. Since the midterm update, we have produced classification results from Support Vector Machine. Also, since then, we have expanded the dataset size to see how the different sizes effects the results.

#### The datasets:
From The Kylberg Texture Dataset [8], we are producing preliminary results from a small collection of the dataset. Instead of looking at all 28 classes to test our implementation of computer vision algorithms, we are training on only on 6 texture classes for the midpoint update and plan to expand to all 28 classes in the final report. There are 40 pictures in each class. Overall, 240 pictures were used. These 6 classes include canvas, cushion, linseeds, sand, seat, and stone. We are looking at images of size 500x500. Additionally, each picture is producing 24 images of varying octaves and scales. For LBP, what is produced as the feature is an arrays of histograms for the different images from the dataset. The histogram has a bin range from 0-255 representing the different unique LBP binary values counted from calculation, and the frequency of these values. Thus, we find that 100x100 is a good starting size to begin using to test and refine our algorithms. 

We then split this training set with 80/20 as fitting/predicting data sets. From the fitting set, we feed each image into one of the four feature detection algorithms (Local Binary Pattern, Scale-Invariant Feature Transform and Gray Level Co-Occurrence Matrix). Then, we pass the array of features from each individual algorithm into the Support Vector Machine to fit the data. Then, we use the predicting data set and predicted the classes from the given data. We then evaluated the results by accuracy, precision and recall.
We use Support Vector Machine because it is often used for classification with feature vectors, and there are prior research examples of SVM being used with the algorithms that are in our proposal [4][14][13][2][7].

#### Hyperparameters

With the exception to SIFT, which can take hours to compute features, to find the optimal values for the hyperparameter values, we used cross validation and grid search. Grid Search takes in the parameters of SVM and then runs an exhaustive search over all combinations of the parameters to find the best combination. Cross Validation is used in attempt to prevent over fitting. A group of k folds is used for testing. In each of those folds data is only compared to k -1 pieces of data [17].  
For the SVM estimator for LBP, the best parameters Grid Search CV recommended were: 

#### Challenges

##### SURF
One of the changes we made since the midterm update is the decision to remove SURF from the list of algorithms we are testing. Since SURF is a less used algorithm it was harder to find resources for implementation. Two separate implementations of surf were attempted based on different online resources[9][18]. Both of our attempts were unsuccessful and we were discouraged from using SURF from a library so we decided that allocating more time this algorithm was not beneficial to the project. Since SURF is based off of SIFT [9] but SIFT, LBP and GLCM are different from each other we feel as though the project is just as effective without testing SURF.
Another challenge that arose during the second half of implementation was the amount of time it took to run the algorithms. Some of our team member’s computers completed computations slower than others so we had to send code to each other to run it. Even with faster computers, some algorithms still took almost 11 hours to run. Due to this we were were not able to calculate all of the results we would have liked. For instance SIFT could not be fully tested, so we have to use the results we did get as a predictor. 

##### SIFT
SIFT proved to be too computationally expensive to run the entire dataset. Each image took, on average, one minute to compute an array of size 128 vectors, and we had about 3000 images to look at for the dataset. This would have taken two days for one machine, but it would not have been a viable option for any of our machines to handle this computation. Thus, We did not get to test the final results on the entire 28 classes dataset. Instead, we looked at a variety of k-clusters and found the best results for the 500x500 dataset. Still, extracting features for 80% of this dataset took on average, about 8 hours. Additionally, testing on SVM required about 3 hours. Thus, in our findings, we find that the time and computational power to use SIFT is a major drawback to the algorithm.

#### The algorithms:

##### SIFT:
SIFT:  We looked at 4 octaves of quartered, halved, normal and double sizes which contributes to the image scale invariant aspect of the algorithm. There are 6 scales or 6 gaussian blurring intensities applied at each octave. As described in David Lowe’s work, each increasing scale is increased by multiplicative factor of k [1][5][15]. The value for k is chosen as 1.41424 as a starting point. For each increasing octave with the same scale level, the sigma is also a multiplicate of k. This process is to ensure that we detect dominant features that are dominant across all scales [1]. Then, the difference-of-gaussian function is used to provide true scale invariance [1]. This produces 5 difference of gaussian scale per octave. Then, local extrema detection is performed. Per octave, for each point in 4 difference of gaussian scale level, 9 neighbors from the previous scale, 9 neighbors from the next scale, and 8 neighbors from the current scale are examined or a total of 26 neighboring scales that are within 1 pixel from the point. If the current pixel is the minimum or maximum of all neighbors, it is chosen as a minimum or maximum point. Additionally, each point must pass a threshold to be examined as a an  minimum or maximum point. Then, for each image across all octaves and scales, orientation and magnitude are found. For each of these extrema points, a keypoint is produced which contains information about the location of the keypoint, its gaussian sigma, and its dominant orientation. Then, descriptors are produced from each keypoint. For each keypoint, a descriptor/feature vector of size 128 is produced. It is essentially a window of size 16x16 pixels surrounding the feature pixel. Then, it is a description of 16 4x4 histograms in which each histogram contains 8 bins, and each represents an orientation of 45 degrees. While the majority of the implementation is cross checked between the following resources including Lowe’s paper and tutorials on SIFT, existing code implementation is used to guide us when critical [15][5]. This is especially important in finding the gaussian window weight to give more priority to orientation that are closest to the feature pixel. Further implementation will include clustering the SIFT features and using a histogram of SIFT features in a bag-of-features model to classify with SVM.

##### Local Binary Pattern
You want to calculate the lbp value for each pixel. Iterating through the grayscale image version of the pic, you set the current pixel as the center pixel for your 3x3 neighborhood. You threshold the the neighborhood by comparing the center pixel values to all the neighbors, and if the center less than the neighbor, set the threshold to one. Else set the value to zero. The LBP value you get is the 8-binary neighborhood of the center pixel converted to a decimal representation. To get this representation, start in a clockwise fashion and the top right neighbor as 2^0 and continue from there. Multiple by the threshold values. That is your LBP value. 
 ![Image](https://i.imgur.com/VDVbcBe.png)
 
 Source for Figure: [16]
 
You then get the 2-Dimensional Image Representation of the LBP. You can use this to calculate a histogram for that image.  Next is constructing the histogram of the frequencies of the LBP values calculated (in percentage). Since the 2-D Since a 3 x 3 neighborhood has 2 ^ 8 = 256 possible patterns, the LBP 2D array thus has a minimum value of 0 and a maximum value of 255,  and so a 256-bin histogram is constructed of LBP codes [3]. Each image will return a histogram, which is just a one dimensional array. You use this as the feature of the images, along with the label set of images from  1-6 for each image that represents the different folders of textures. I then used SVC, and then used a test set of the histograms and labels, to see the predicted labels that the model would get. And then compared the original labels and the predicted labels. 


##### Gray Level Co-Occurrence Matrix
In this midterm update, a dataset of 6 textures with 40 samples in each class is used to obtain GLCM values and corresponding statistical metrics. The GLCM is a n x n matrix where n is the number of gray level in an image and each cell represents the co-occurrence of gray-level pixel in an image. Thus, all image samples are converted to 8 bit grayscale images and a GLCM is calculated for each converted image. Multiple GLCM calculations are done with displacement vector of [1, 2, 4] and rotation vector of [0] to find the optimal combination for statistics calculations. 3 possible sets of GLCMs are normalized by the number of pixels and then used to calculate the values of contrast, dissimilarity, homogeneity, ASM, energy, and correlation of the texture image by following equations. [7]

 ![Image](https://i.imgur.com/Pfo51Fc.png)

After getting all values, a training set and test set for SVM model are constructed with above statistics as attributes and 6 textures: canvas, cushion, linseed, sand, seat, and stone as labels. The dataset is divided into a training set and a test set with ratio of 8:2, 7:3, and 6:4 and feeded to three different SVM implemented in Python - sklearn libraries. The result models are sensitive to input images, hyperparameters, and SVM algorithms. To maximize the overall accuracy of the model, three types of SVM, SVC, linearSVC, and NuSVC, are run with different parameters settings such as penalty value, kernel type, and gamma value. Also, the original images and resized images are tested to check the influence of different image input.

### Experiments and results

#### SIFT

 ![Image](https://i.imgur.com/4vqPfgd.png)
 
 ![Image](https://i.imgur.com/TvDEnM5.png) 
 
 ![Image](https://i.imgur.com/Po6KDPG.png)
 
 ![Image](https://i.imgur.com/XNHTPxJ.png)
 
Above are the images of features extracted of varying octaves and scales. There are 4 octaves and 6 scales. The rate that gaussian sigma is multiplied is k = 1.414214. The starting sigma is .7. For the first two octaves, no features were found throughout all the scales. Only a few features were found in the normal size octave and most were found in the doubled sized octave. In the midterm, we looked at images of size 100x100. In the final update, we looked at performance of images of size 500x500.

 ![Image](https://i.imgur.com/Sn149Et.png)

Here, the results show the baseline of only passing in an array of descriptors into SVM to fit and predict. We have roughly 50% accuracy rate for all data splits. The precision and recall scores are very similar as well. This shows that on average, relevant classes are correctly identified at around 50% of the time (recall). On average for all classes, 50% of those identified for a class are correct (precision). 
The results are what we expected if a bag-of-feature model is not employed, thus these are the baseline scores. Instead of passing in the array of descriptors, we must first cluster the features with K-means. Then, map the label to cluster for descriptors in each image. Each image will have a histogram of how often these descriptors show in the image. Then, an array of these histogram features is used to fit and predict the classification for these images.

 ![Image](https://i.imgur.com/K9WyNla.png)
 ![Image](https://i.imgur.com/sHauWrO.png)
 
 The following are the results for properly classifying using the bag-of-words model on a partial dataset with only 6 classes. First, we used SIFT to extract the features from the images. Then, we clustered these features that were collected from images in the training set. Then, the bag-of-word histograms were created with k-mean closest clusters. Then, these histograms were used as prediction for SVM classifier. Then, the classifier produced these results on the test set. 
Here, we see that varying the k-values has important effects on the outcomes. The best performance resulted from using 20 clusters with less than 80% precision, recall, and accuracy. Overall, this method produces better results than the baseline with over 60% accuracy for all k values. Further, the performance improved with a larger starting image to start from. In Table 1, we see that the best accuracy result is 50% with a 100x100 as the starting size for k=10. In Table 2, we see that the accuracy improved to 60% with a starting size of 500x500.
However, this method is very computationally expensive. Combining SIFT, k-means clustering, and SVM resulted in hours just to compute one run for each k-clusters. Each run took about 8 hours to complete just to find all SIFT features. Therefore, we did not get to compute one run of the entire dataset with 28 classes, which would have taken us more than 50 hours on a regular machine. Thus, we did not get to run this algorithm on the full data-set. However, when viewing results from other algorithms such as LIB and GLCM, which had lower accuracy results on the entire dataset, it is likely that SIFT would have poorer performance on the full dataset as well. It is the conclusion that our SIFT implementation has the highest performance on of less than 80% accuracy on the texture dataset of size 500x500.


#### LBP

2D LBP representation of different image texture.

 ![Image](https://i.imgur.com/BgpiFCj.png)

Here are the corresponding histograms for these images:
![Image](https://i.imgur.com/PXxNy1A.png) 
![Image](https://i.imgur.com/aIz3zdz.png) 
![Image](https://i.imgur.com/2g5PI4S.png) 
![Image](https://i.imgur.com/vCzDcC4.png)


SVM Results for LBP:

![Image](https://i.imgur.com/Yl1C1q8.png)

The results show very high accuracy and precision, and we see the accuracy increase as the training set increases and then decrease. This is a key sign of overfitting, which is causing such high results and the accuracy to plateau and then decrease as the training set increases. Future work would be to see why this overfitting is occurring, and using cross validation to combat it. 

### GLCM 


![Image](https://i.imgur.com/2czPsg5.png)

![Image](https://i.imgur.com/U8Hrdi0.png) 

It can be noticed that GLCM algorithm does not extract distinctive features when the resized images are used. Therefore, only images with original size are used to make a SVM classifier.

SVM Results for GLCM

Altering displacement vector in linear kernel type

![Image](https://i.imgur.com/JoxDsMw.png) 

Since the result of LinearSVC is too low, it is omitted and only SVC and NuSVC are used for further optimization.

![Image](https://i.imgur.com/juODoeB.png)

![Image](https://i.imgur.com/kBikm4p.png)

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

In the midterm update, we stated that our goals for the final project were to execute all four algorithms. However, one of the challenges we faced during the second half of this project was fully implementing the SURF method. We ultimately decided to remove it from our project and instead spend our time doing a proper comparison of the other three algorithms Local Binary Pattern (LBP),  Scale-Invariant Feature Transform (SIFT), and Gray Level Co-Occurrence Matrix (GLCM). Additionally, one of our goals is to run SVM with bag-of-model on the full data-set, but we were limited by computational power to only look into 6 classes of 40 images in each class with 500x500 dimensions. To extract features on 80% of the dataset required an average of 8 hours of time. In addition, it took 3 hours to get the final prediction SVM results. Still the SIFT results in this paper can be used as predictors of what the results would be on the full dataset, which we would predict would be lower.

The overall result shows that the most preferable algorithm to use to classify is of Local Binary Pattern (LBP) over Scale-Invariant Feature Transform (SIFT) and Gray Level Co-Occurrence Matrix (GLCM). The algorithm produced the highest accuracy rate with the least amount of time. Next, SIFT showed promising results but proved to be too computationally expensive to run. Due to the feature extraction process which involved creating many scaled-versions of the image and blurring-versions of image, SIFT involves a lot of memory space and processing power. The algorithm showed improvement on larger scale images as a starting point, but this would involve more memory to work with these images. We were able to complete bag-of-words model with SVM on a partial dataset with images of size 500x500 for 6 classes but not the entire 28 classes due to computation limitations on our local machines.


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

[17] Cross-validation: evaluating estimator performance Retrieved From  					https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation 

[18] OpenSURF Retrieved From  		       	…….https://www.mathworks.com/matlabcentral/fileexchange/28300-opensurf-including-image-warp 
