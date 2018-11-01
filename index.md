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

 ![Image](https://imgur.com/Sn149Et)

Here, the baseline of only passing in an array of descriptors into SVM to fit and predict. We have roughly 50% accuracy rate for all data splits. The precision and recall scores are very similar as well. This shows that on average, relevant classes are correctly identified at around 50% of the time (recall). On average for all classes, 50% of those identified for a class are correct (precision). 

The results are what we expected if a bag-of-feature model is not employed, thus these are the baseline scores. Instead of passing in the array of descriptors, we must first cluster the features with K-means. Then, map the label to cluster for descriptors in each image. Each image will have a histogram of how often these descriptors show in the image. Then, an array of these histogram features is used to fit and predict the classification for these images.

#### LBP
 ![Image](https://i.imgur.com/BgpiFCj.png)

Here are the corresponding histograms for these images:
![Image](https://i.imgur.com/BgpiFCj.png)
![Image](https://i.imgur.com/BgpiFCj.png)
![Image](https://i.imgur.com/BgpiFCj.png)
![Image](https://i.imgur.com/BgpiFCj.png)

SVM Results for LBP:
![Image](https://i.imgur.com/Yl1C1q8.png)

The results show very high accuracy and precision, and we see the accuracy increase as the training set increases and then decrease. This is a key sign of overfitting, which is causing such high results and the accuracy to plateau and then decrease as the training set increases. Future work would be to see why this overfitting is occurring, and using cross validation to combat it. 

#### SURF
2D LBP representation of different image texture


### GLCM 

### Conclusion

### References

[1] Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision,60(2), 91-110. [https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

[2] Ojala, T., Pietikäinen, M., & Mäenpää, T. (2001). A Generalized Local Binary Pattern Operator for Multiresolution Gray Scale and Rotation Invariant Texture Classification. Lecture Notes in Computer Science Advances in Pattern Recognition — ICAPR 2001,399-408. doi:10.1007/3-540-44732-6_41

[3] Local Binary Patterns with Python & OpenCV. (2018, June 21). Retrieved from [https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/](https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)

[4] Pietikã-Inen, M. (2010). Local Binary Patterns. Scholarpedia,5(3), 9775. doi:10.4249/scholarpedia.9775

[5] Sinha, U. (n.d.). Generating a feature. Retrieved from [http://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/)

[6] Texture. (n.d.). Retrieved from [http://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.pdf](http://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.pdf)

[7] Renzetti, L. Z. (2011). Use of a Gray Level Co-occurrence Matrix to Characterize Duplex Stainless Steel Phases Microstructure. Frattura ed Integrità Strutturale, 16, 43-51. doi:10.3221/IGF-ESIS.16.05 http://www.gruppofrattura.it/pdf/rivista/numero16/numero%2016%20articolo%205.pdf

[8]Corbett-Davies, S. Real-World Material Recognition for Scene Understanding.

[9] 1.4. Support Vector Machines. (n.d.). Retrieved from [http://scikit-learn.org/stable/modules/svm.html](http://scikit-learn.org/stable/modules/svm.html)

[10] Sklearn.cluster.KMeans. (n.d.). Retrieved from [http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

[11] G. Kylberg. The Kylberg Texture Dataset v. 1.0, Centre for Image Analysis, Swedish University of Agricultural Sciences and Uppsala University, External report (Blue series) No. 35. Available online at: [http://www.cb.uu.se/~gustaf/texture/](http://www.cb.uu.se/~gustaf/texture/)

[12] Classification: Precision and Recall Machine Learning Crash Course Google Developers. (n.d.). Retrieved October 10, 2018, from [https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)

[13] Classification: Accuracy Machine Learning Crash Course Google Developers. (n.d.). Retrieved from [https://developers.google.com/machine-learning/crash-course/classification/accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy)

[14] Pederson, J. T. (2011).Study group SURF: Feature detection & description [PDF]. Aarhus: Aarhus University. [http://cs.au.dk/~jtp/SURF/report.pdf](http://cs.au.dk/~jtp/SURF/report.pdf)

[15 ] Chi Chung Tam, D. (2010). SURF: Speeded Up Robust Features [PDF]. Toronto: Ryerson University. [http://www.computerrobotvision.org/2010/tutorial_day/tam_surf_rev3.pdf](http://www.computerrobotvision.org/2010/tutorial_day/tam_surf_rev3.pdf)

[16] Birchfield, S. (n.d.). SURF detectors and descriptors [PPT]. Clemson: Clemson University.
