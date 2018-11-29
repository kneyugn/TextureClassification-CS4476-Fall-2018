## Classifying Textures: Final Update

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

For  GLCM, the recommended values for the parameters were : 
![Image](https://i.imgur.com/kkHNriF.png)

#### Challenges

##### SURF
One of the changes we made since the midterm update is the decision to remove SURF from the list of algorithms we are testing. Since SURF is a less used algorithm it was harder to find resources for implementation. Two separate implementations of surf were attempted based on different online resources[9][18]. Both of our attempts were unsuccessful and we were discouraged from using SURF from a library so we decided that allocating more time this algorithm was not beneficial to the project. Since SURF is based off of SIFT [9] but SIFT, LBP and GLCM are different from each other we feel as though the project is just as effective without testing SURF.
Another challenge that arose during the second half of implementation was the amount of time it took to run the algorithms. Some of our team member’s computers completed computations slower than others so we had to send code to each other to run it. Even with faster computers, some algorithms still took almost 11 hours to run. Due to this we were were not able to calculate all of the results we would have liked. For instance SIFT could not be fully tested, so we have to use the results we did get as a predictor. 

##### SIFT
SIFT proved to be too computationally expensive to run the entire dataset. Each image took, on average, one minute to compute an array of size 128 vectors, and we had about 3000 images to look at for the dataset. This would have taken two days for one machine, but it would not have been a viable option for any of our machines to handle this computation. Thus, We did not get to test the final results on the entire 28 classes dataset. Instead, we looked at a variety of k-clusters and found the best results for the 500x500 dataset. Still, extracting features for 80% of this dataset took on average, about 8 hours. Additionally, testing on SVM required about 3 hours. Thus, in our findings, we find that the time and computational power to use SIFT is a major drawback to the algorithm.

##### GLCM
The performance of SVM based on GLCM features can be noticeably decreased when the extracted features from GLCM algorithm are not significant or distinguishable. From the 28 textures, 17 out of 28 classes fails to generate correct GLCMs and only 11 classes are considered as useful data to make a SVM model. It is hard to find optimal distance vector and rotation vector for GLCM calculation. Thus, even after several manual parameter tests, GLCM algorithm cannot compute proper GLCMs and attributes for SVM. Also, the accuracy of the model is lower than the results from other algorithms. Although the computation cost is cheap for using GLCM algorithm, the model does not strongly represent the entire texture dataset.

#### The algorithms:

##### SIFT:
SIFT:  We looked at 4 octaves of quartered, halved, normal and double sizes which contributes to the image scale invariant aspect of the algorithm. There are 6 scales or 6 gaussian blurring intensities applied at each octave. As described in David Lowe’s work, each increasing scale is increased by multiplicative factor of k [1][5][15]. The value for k is chosen as 1.41424 as a starting point. For each increasing octave with the same scale level, the sigma is also a multiplicate of k. This process is to ensure that we detect dominant features that are dominant across all scales [1]. Then, the difference-of-gaussian function is used to provide true scale invariance [1]. This produces 5 difference of gaussian scale per octave. Then, local extrema detection is performed. Per octave, for each point in 4 difference of gaussian scale level, 9 neighbors from the previous scale, 9 neighbors from the next scale, and 8 neighbors from the current scale are examined or a total of 26 neighboring scales that are within 1 pixel from the point. If the current pixel is the minimum or maximum of all neighbors, it is chosen as a minimum or maximum point. Additionally, each point must pass a threshold to be examined as a an  minimum or maximum point. Then, for each image across all octaves and scales, orientation and magnitude are found. For each of these extrema points, a keypoint is produced which contains information about the location of the keypoint, its gaussian sigma, and its dominant orientation. Then, descriptors are produced from each keypoint. For each keypoint, a descriptor/feature vector of size 128 is produced. It is essentially a window of size 16x16 pixels surrounding the feature pixel. Then, it is a description of 16 4x4 histograms in which each histogram contains 8 bins, and each represents an orientation of 45 degrees. While the majority of the implementation is cross checked between the following resources including Lowe’s paper and tutorials on SIFT, existing code implementation is used to guide us when critical [15][5]. This is especially important in finding the gaussian window weight to give more priority to orientation that are closest to the feature pixel. Further implementation will include clustering the SIFT features and using a histogram of SIFT features in a bag-of-features model to classify with SVM.

##### Local Binary Pattern
You want to calculate the lbp value for each pixel. Iterating through the grayscale image version of the pic, you set the current pixel as the center pixel for your 3x3 neighborhood. You threshold the the neighborhood by comparing the center pixel values to all the neighbors, and if the center less than the neighbor, set the threshold to one. Else set the value to zero. The LBP value you get is the 8-binary neighborhood of the center pixel converted to a decimal representation. To get this representation, start in a clockwise fashion and the top right neighbor as 2^0 and continue from there. Multiple by the threshold values. That is your LBP value. 
 ![Image](https://i.imgur.com/VDVbcBe.png)
 
 Source for Figure: [16]
 
You then get the 2-Dimensional Image Representation of the LBP. You can use this to calculate a histogram for that image.  Next is constructing the histogram of the frequencies of the LBP values calculated (in percentage). Since the 2-D Since a 3 x 3 neighborhood has 2 ^ 8 = 256 possible patterns, the LBP 2D array thus has a minimum value of 0 and a maximum value of 255,  and so a 256-bin histogram is constructed of LBP codes [3]. Each image will return a histogram, which is just a one dimensional array. You use this as the feature of the images, along with the label set of images from  1-6 for each image that represents the different folders of textures. I then used SVC, and then used a test set of the histograms and labels, to see the predicted labels that the model would get. And then compared the original labels and the predicted labels. 


##### Gray Level Co-Occurrence Matrix
For the GLCM experiment, a dataset of 28 textures with 40 to 160 images in each class is used to calculate GLCM and corresponding statistical metrics. The GLCM is a n x n matrix where n is the number of gray level in an image and each cell represents the co-occurrence of gray-level pixel in an image. Thus, all image samples are converted to 8 bit grayscale images and a GLCM is calculated from each converted image. After several tests, the parameters for GLCM computation are set to displacement vector of (1) and rotation vector of (0). Then, with these GLCMs, 6 GLCM properties - contrast, dissimilarity, homogeneity, ASM, energy, and correlation are calculated with the following equations. [7]


 ![Image](https://i.imgur.com/Pfo51Fc.png)

After getting all values, a training set and test set for the support vector machine model are constructed with above statistics as attributes and 28 textures as labels. The dataset is divided into a training set and a test set with the ratio of 7:3 and SVC model is generated with sklearn.svm library in Python environment. To maximize the overall accuracy of the model, hyperparameters are obtained and the cross validation is processed from GridSearchCV function in sklearn.model_selection. Based on the optimized hyperparameters, poly kernel type is used with 1 penalty value, degree 6, shrinking and probability features, one-vs-one decision function, ⅙ gamma, 5.0 coefficient and unlimited iteration. Also, the original images of 576 x 576 size and resized images of 100 x 100 size are tested to check the influence of different scaled inputs.


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

However, this method is very computationally expensive. Combining SIFT, k-means clustering, and SVM resulted in hours just to compute one run for each k-clusters. Each run took about 8 hours to complete just to find all SIFT features. Therefore, we did not get to compute one run of the entire dataset with 28 classes, which would have taken us more than 50 hours on a regular machine. Thus, we did not get to run this algorithm on the full data-set. However, when viewing results from other algorithms such as LBP and GLCOM, which had lower accuracy results on the entire dataset, it is likely that SIFT would have poorer performance on the full dataset as well. It is the conclusion that our SIFT implementation has the highest performance on of less than 80% accuracy on the texture dataset of size 500x500.


#### LBP

2D LBP representation of different image texture.

 ![Image](https://i.imgur.com/XylPErI.png)
 ![Image](https://i.imgur.com/PsCJlWF.png)
 ![Image](https://i.imgur.com/6zebMjV.png)
 ![Image](https://i.imgur.com/Aj3R6Lw.png)
 ![Image](https://i.imgur.com/QTiTw0C.png)
 

Here are the corresponding histograms for these images:
![Image](https://i.imgur.com/Kdvsyje.png) 
![Image](https://i.imgur.com/xXtwisa.png) 
![Image](https://i.imgur.com/q4IaSQm.png) 
![Image](https://i.imgur.com/8uJh2Ad.png)
![Image](https://i.imgur.com/rhZnIRg.png)


SVM Results for LBP:

For the last update, we  saw that for LBP the results were very high when only using the partial dataset. This was a sign of overfitting, as the training and testing error were converging very fast when increasing training set size, with the training error dropping drastically as the training set size increased. Overfitting was occuring because training and testing set are using the same data and fitting to noise and outliers. This was evident when using the sample set below. 

![Image](https://i.imgur.com/yr5aoHj.png)

When looking at the sample dataset, when the degree reached 7 for using sample set and 100x100 images, the accuracy became 100%. Because there are so little classes 6 of them, the algorithm figured out quickly how to spot them, and thus started overfitting. For the sample set that used 576x576 images, the accuracy for both kernel degree 6 and 7 were 100%. Overfitting was solved when using the full dataset, as there are 28 classes to try to identify, versus 6. 

![Image](https://i.imgur.com/lkbGzjL.png)

![Image](https://i.imgur.com/CpQlo3l.png)

![Image](https://i.imgur.com/LbQyClK.png)

![Image](https://i.imgur.com/h5hQlMB.png)

When looking at the results of using the full dataset, they were pretty good. In fact, the accuracies of using a patch of the image 100x100, versus the full 576x576 did not have much difference in  accuracies. For Kernel Degree = 7 100x100 the accuracy was 97.2 versus for Kernel Degree = 7 576x576 97.6%. The accuracies were pretty similar, but the run time for using the full time image was far longer to compute the LBP of the image, 2 hours vs 15 minutes for degree = 7. Kernel of 6 for 100x100 versus 576x576 had a difference, but that is again not the parameter we got from hypertuning. Thus using a patch of the image does a good job of classification, and is an efficient run time. 

When comparing the runtimes of the different kernels, degree 6 was faster every time, but the accuracies were lower compared to degree 7. This makes sense, as the complexity of the algorithm increases with each addition of degree to the kernel. 



### GLCM 

40 to 160 images of 28 textures with size 100 x 100 and 576 x 576 are converted to GLCM with 1 distance vector and 0 angle vector. Then, 6 GLCM properties, contrast, dissimilarity, homogeneity, energy, correlation and ASM are calculated. Two examples of calculated statistics data are provided below.

Sample training set of GLCM features with 28 images of 100 x 100 size

![Image](https://i.imgur.com/wqw2buw.png)

Sample training set of GLCM features with 28 images of 576 x 576 size

![Image](https://i.imgur.com/c1WDlbR.png)

SVM Results for GLCM

![Image](https://i.imgur.com/1VMZWxx.png)

This table shows the performance of SVM using features obtained from GLCM with the optimized hyperparameters. Clearly, the results with bigger size images is better than smaller size. However, as shown in the table of calculated GLCM properties, some textures have no distinctive statistical values such as ceiling2, cushion1, and floor1. Therefore, the overall accuracy of the classification is low as 30% for 576 x 576 size images and 20% for 100 x 100 size images.

Additional SVM Experiment with Meaningful Classes

In spite of the previous result, the performance of this GLCM classification can be improved by removing texture classes with insignificant GLCM properties. Total 17 classes are eliminated and 11 classes are used to create new svm model. The overall accuracy is significantly increased to 60% which is two times better than the previous experiment. The sample calculation and the experiment result are in the next tables.

Sample training set of GLCM features with significant 11 images of 576 x 576 size

![Image](https://i.imgur.com/VipNlzc.png)

![Image](https://i.imgur.com/bnD58Pa.png)

![Image](https://i.imgur.com/GhmfthM.png)


### Conclusion

In the midterm update, we stated that our goals for the final project were to execute all four algorithms. However, one of the challenges we faced during the second half of this project was fully implementing the SURF method. We ultimately decided to remove it from our project and instead spend our time doing a proper comparison of the other three algorithms Local Binary Pattern (LBP),  Scale-Invariant Feature Transform (SIFT), and Gray Level Co-Occurrence Matrix (GLCM). Additionally, one of our goals is to run SVM with bag-of-model on the full data-set, but we were limited by computational power to only look into 6 classes of 40 images in each class with 576x576 dimensions. To extract features on 80% of the dataset required an average of 8 hours of time. In addition, it took 3 hours to get the final prediction SVM results. Still the SIFT results in this paper can be used as predictors of what the results would be on the full dataset, which we would predict would be lower.

The overall result shows that the most preferable algorithm to use to classify is of Local Binary Pattern (LBP) over Scale-Invariant Feature Transform (SIFT) and Gray Level Co-Occurrence Matrix (GLCM). The algorithm produced the highest accuracy rate with the least amount of time. When using the full dataset, the accuracy we saw was 97.7% which was pretty impressive. The run time was around 2 hours, much faster than SIFT, but slower than GLCM. However the trade off for the run time for GLCM is the low accuracies that it produced. But when using only a patch of the image 100x100 for LBP, the run time was faster of 15 minutes, which can be comparable with GLCM, and produced an accuracy of 97.2%. Thus, when looking at both run time efficiency and classification accuracy, we conclude LBP is the best for classifying textures.   Next, SIFT showed promising results but proved to be too computationally expensive to run. Due to the feature extraction process which involved creating many scaled-versions of the image and blurring-versions of image, SIFT involves a lot of memory space and processing power. The algorithm showed improvement on larger scale images as a starting point, but this would involve more memory to work with these images. We were able to complete bag-of-words model with SVM on a partial dataset with images of size 576x576 for 6 classes but not the entire 28 classes due to computation limitations on our local machines. Finally, GLCM algorithm results the lowest performance among all three algorithms. Even though calculating GLCMs and training obtained features with SVM take less than a minute, more than half of calculated features are not useful to represent input textures. Therefore, the overall score for entire model is decreased to 30%. Removing unrepresentative texture classes raises the accuracy of the model, but it is only 60% which is lower than other models’ results. 

It is interesting also to compare the different features that were generated from each of these algorithms. The feature vector generated by LBP is an array of the percent count of each of  the 256-bin histogram of LBP codes. This proved to be a robust feature space, as each texture had a unique LBP histogram, and  was helpful in identifying the differences. It also computed fast, as even though the final feature vector for each image has 256 values, they all are generated relatively quickly from the histogram array. As for SIFT, the each feature is a vector of size 128. Each value in the vector is the dominant orientation within a 4x4 window histogram. Each image can produce about 30 or more vectors among the different octave and scale images. Thus, this contributed to the slow computation time, which is a major drawback of using SIFT as a classifier. Unlike the other algorithms, GLCM method produces small amount of features as 6 statistical values for each texture image. Due to this small size features space, the performance of SVM using GLCM algorithm is highly sensitive to the calculated values and failing to obtain appropriate values leads to the low accuracy.


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
