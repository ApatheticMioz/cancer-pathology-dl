#### FACE-SWAP DEEPFAKES DETECTION

#### USING NOVEL MULTI-DIRECTIONAL

#### HEXADECIMAL FEATURE DESCRIPTOR

###### PRESENTER

###### QURAT-UL-AIN, PHD SCHOLAR( UET TAXILA)

###### RESEARCH ASSISTANT

###### MSP LAB

###### SOFTWARE ENGINEERING DEPARTMENT

###### UNIVERSITY OF ENGINEERING AND TECHNOLOGY-TAXILA, PAKISTAN

##### 16 th August 2022^19

##### th IBCAST 2022


# Multimedia Signal Processing Lab

**2**

```
MSP research lab focuses on generating cutting-edge research results in the field of
multimedia signal processing. MSP is conducting useful research and designs various
state-of-the-art computer vision, image and audio processing based applications by
using both learning and non-learning-based techniques. The specific application areas
associated with research in our lab include:
 Audio Processing: Audio stream analysis for excitement detection, speech and music
segregation, speech recognition, audio forensics.
 Video Processing: Video summarization, DeepFakes, surveillance, video analytics,
understanding, segmentation, object tracking and recognition, content-based
retrieval, activity and gesture recognition, multimedia applications.
 Image Processing: Brain Aneurysm Detection and Rupture Prediction, Morphological
image processing, tracking and recognition, segmentation, computer vision, image
processing for biometrics.
```

(^3) Presentation Outline
❑ Introduction (Background and Motivation)
❑ Problem Statement (Challenges in state-of-art descriptors)
❑ Aims and Objectives
❑ Literature Work
❑ Contributions
❑ Proposed Methodology(face-swap deepfakes deketection)
❑ Performance Evaluation
❑ Conclusion


# Introduction

**4**

```
❑ The growing number of sophisticated deep
learning algorithms lead in creation of highly
realistic deepfake videos.
```
```
❑ face-swap is the most commonly employed
identity swapping deepfakes approach.
```
```
❑ The motivation is to develop a robust local
texture descriptor to extract more directional
and magnitude details from face-swap
deepfake detection.
```
```
Fig.1:Original frames
```
```
Fig.2: GAN generated face-swap frames
```

### Problem Statement: Challenges in

### Face-swap Deepfake Detection

**5**

```
❑ Existing descriptors are failed to
detect videos containing variations of
the facial skin tone of people having
different races, illumination conditions,
presence of accessories like glasses
on the face, and loss of details due to
compressed video resolution.
```
```
❑ Methods based on local texture
descriptors [ 17 ] like LBP, SURF, and
LTP compute only limited directional
information and disregard the
magnitude information.
Fig.3: Frames form datasets containing people of different
skin color, Illumination conditions, and faces with glasses.
```

# Aims and Objectives

**6**

❑ The major objectives of the proposed research work are:

```
✓ Our motivation was to develop a robust descriptor by capturing more
directional and higher-magnitude details from the adjacent pixels to
effectively represent the frames of real and face-swap videos.
```
```
✓ A multi-directional feature descriptor, which effectively captures the
texture orientation and magnitude information from the video frames.
```
```
✓ A deepfakes detection system that is robust on videos containing variations
of the facial skin tone of people having different races, and illumination
conditions.
```

# Problems in Existing Literature

**7**

```
Methods Technique Problems
[ 9 ] SVM Classifier+
landmarks
```
Fail to detect blur images.

```
[12] SVM classifier Degrade performance on person off-
looking frames.
[15] Texture energy-based
features + (MLP, LogReg) Fail to detect faces with closed eyes.
[8] SURF + SVM Not robust on unseen data.
[13] EAR+ landmarks Fail to detect video frame with more
eye blinking scenarios.
[21] Deep Features + CNN Not robust on detection of unseen
videos.
[22] Deep Features + SVM Performance fails on compressed
videos.
```
```
Table 1. LITERATURE REVIEW OF CONTEMPORARY METHODS.
```

# Contributions

**8**

```
The main contributions of the proposed work are:
```
```
✓ We propose a novel multi-directional hexadecimal feature descriptor (MDHFD) to
effectively capture the texture orientation and magnitude information from the video
frames.
```
```
✓ Our proposed face-swap deepfakes detection system is robust on videos containing
variations of the facial skin tone of people having different races, illumination
conditions, presence of accessories like glasses on the face, and loss of details due to
compressed video resolution.
```
```
✓ Rigorous experimentation was performed on two diverse deepfakes datasets
including the cross-corpora evaluation to test the generalizability of our method.
```

### Face-swap Deepfakes Detection

### Framework

**9**

```
Fig.4: Proposed method
```
**_Multi-directional hexadecimal feature descriptor_**


**10**

❑ **Face Detection**

```
✓ Multi-task Cascaded Convolutional Networks (MTCNN)
[ 18 ] is used for frame-level extraction.
```
```
✓ MTCNN detect and extract only the face portion from
the input video.
```
```
✓ In comparison with other face detectors like Haar
Cascade, MTCNN detects the faces precisely in the
presence of occlusion and varying illumination
conditions.
```
# Face-swap Deepfakes Detection

# Framework


**11**

# Face-swap Deepfakes Detection

# Framework

❑ **Feature extraction**

```
▪ Multi-directional hexadecimal feature descriptor
```
```
✓ The proposed descriptor is comprised of the direction
and magnitude-based features.
```
```
✓ The orientation patterns effectively extract
discriminative information from the frames and
compute directional information.
```
```
✓ Furthermore, we compute the patterns based on
magnitude which captures additional useful
information.
```
```
✓ Each resultant feature vector is a combination of
magnitude and orientation patterns.
```

**12**

❑ **LHeXDP**

```
✓ For directional information, we employed a local
hexadecimal feature descriptor (LHeXDP).
✓ For a given frame F ( x, y ) , we compute the 1 st order
derivative at the grayscale value of the surrounding pixels
along with directions, as shown in ( 1 ).
```
```
𝐹𝛼^1 (𝑑ℎ,𝑑,𝑣,𝑑𝑏)|𝛼 = 00 , 450 , 900 , 1350 (1)
```
```
✓ The frame is translated into 16 different values from which
the directions are determined.
```
```
✓ We construct texture orientation for each pixel by
comparing the 1 st order derivatives of the center pixel
direction with the directions of all the eight surrounding
neighbors.
```
# Face-swap Deepfakes Detection

# Framework


**13**

❑ **LHeXDP**

```
✓ Taking the 2 nd order derivative of the central pixel, we
obtained 8 - bit directions with all the eight neighbouring
directions (s= 1 - 8 ) using the ( 2 ) and ( 3 ).
```
```
𝐿𝐻𝑋𝐷𝑃^2 =
```
```
𝑇 1 𝐹𝑑𝑖𝑟^1. 𝑑𝑐 .𝐹𝑑𝑖𝑟^1. 𝑑 1 .𝑇 1 𝐹𝑑𝑖𝑟^1. 𝑑𝑐 .𝐹𝑑𝑖𝑟^1. 𝑑 2 ...
...,. 𝑇 1 𝐹𝑑𝑖𝑟^1. 𝑑𝑐 .𝐹𝑑𝑖𝑟^1. 𝑑𝑠 |𝑠= 8
```
```
2
```
```
𝐹𝑑𝑖𝑟^1 𝑑𝑐 ×𝐹𝑑𝑖𝑟^1 𝑑𝑠 =൝^0 , 𝐹𝑑𝑖𝑟
```
(^1). 𝑑𝑐 =𝐹𝑑𝑖𝑟 (^1). 𝑑𝑠
𝐹𝑑𝑖𝑟^1. 𝑑𝑠, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒^ (3)
✓ The 2 nd order is distributed into 16 binary patterns
computed as shown in ( 4 ).
𝐿𝐻𝑋𝐷𝑃^2 |𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛𝑠= 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 15 , 16
=෍
𝑠= 0
𝑠
2 𝑠−^1 𝑥 𝑇 1 (𝐿𝐻𝑋𝐷𝑃^2 𝑑𝑐 )|𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛𝑠= ∝ ( 4 )

# Face-swap Deepfakes Detection

# Framework


**14**

```
❑ LANMP
✓ Local Adjacent Neighborhood Magnitude Pattern
(LANMP) compute 1 st order derivatives that captures
edge information of each frame in detailed.
𝑀 11. 𝑑𝑠 = 𝐹 010 𝑑𝑠^2 + 𝐹 4510 𝑑𝑠^2 + 𝐹`^11050 𝑑𝑠^2 + 𝐹 1351 0 𝑑𝑠^2 ( 5 )
```
```
✓ In ( 7 ), R is the function that discriminates between the
magnitude of surrounding neighbors’ pixels and the
center pixel based on parameter x.
```
```
𝐿𝐴𝑁𝑀𝑃^2 =෍
𝑠= 1
```
```
𝑠
2 𝑠−^1 ×𝑅 (𝑀 11. 𝑑𝑠 −𝑀 11. 𝑑𝑐 )|𝑠= 8 ( 6 )
```
```
𝑅 𝑥 =ቊ^10 ,,^ 𝑥𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒>^ =^0 (7)
```
# Face-swap Deepfakes Detection

# Framework


**15**

# Face-swap Deepfakes Detection

# Framework

❑ **Feature Fusion**

```
✓ After extracting the local patterns (LHeXDP and
LANMP) of each frame, we obtain the histogram
of both patterns.
```
```
✓ The proposed MDHDF is formed by fused
histograms of LHXDP and LANMP.
```
```
Hist𝑀𝐷𝐻𝐹𝐷= Hist𝐿𝐻𝑋𝐷𝑃||Hist𝐿𝐴𝑁𝑀𝑃 (8)
```

```
 Face Forensics++ dataset (FF++) [22].  World Leaders dataset (WLDR) [12].
```
**16**

## Performance Evaluation

```
DATASETS
We evaluated the performance of the proposed method on the original and face-
swapped subset of following two datasets.
```
```
Fig.5: Samples from FF++ Dataset,
Top row contain original and bottom row contain Faceswap
sample.
```
```
Fig.6: Samples from WLDR Dataset,
Top row contain original and bottom row contain Faceswap
sample.
```

**17**

```
❑ Performance Evaluation of proposed method
```
```
✓ To evaluate the effectiveness of our method for face-swap deepfakes detection, we
designed an experiment to evaluate the performance of our method on the original
and face swap subset on both the FF++ and the WLDR datasets.
✓ We resized all frames to 300 × 300 and extracted the features using our MDHFD
for both the training and testing set.
✓ These features are then used to train the SVM classifier for video deepfakes
detection. SVM produce a better result on proposed MDHFD features using a
gaussian kernel.
✓ More specifically, we achieved an accuracy of 92. 3 % and AUC of 0. 99 on the FF++
dataset, whereas achieved an accuracy of 96. 9 % and AUC of 1. 00 on the WLDR
dataset.
```
## Performance Evaluation


**18**

```
Fig.7. Confusion Matrix of a Proposed Method on WLDR
dataset.
Fig.8. Confusion Matrix of a Proposed Method on
FF++ dataset.
```
```
❑ Confusion matrix Analysis
✓ To better investigate the false-acceptance and rejection scenarios, we designed a
confusion matrix analysis to depict the classification performance of our method on
both datasets as shown below.
```
## Performance Evaluation


**19**

```
Fig.9. ROC of a Proposed Method on WLDR
dataset.
```
```
Fig.10. ROC of a Proposed Method on FF++
dataset.
```
```
❑ ROC Curve Analysis
✓ ROC curve analysis is performed in our third experiment to illustrate the
performance of the proposed method for face-swap deepfake detection.
✓ Using SVM with proposed features we achieved the best AUC of 0. 99 on the FF++
dataset. On the WLDR dataset, we achieved the best AUC of 1. 00 with SVM.
```
## Performance Evaluation


**20**

```
❑ Performance comparison with other classifiers
```
```
Classifiers
```
```
FF++ Dataset WLDR Dataset
```
```
Accuracy AUC Accuracy AUC
SVM 92. 3 % 0. 99 96. 9 % 1. 00
Bagged Trees Ensemble 89. 9 % 0. 97 95. 6 % 1. 00
```
```
Narrow Neural Network 85. 49 % 0. 94 95. 5 % 1. 00
Fine Trees 77. 3 %. 0. 82 91. 9 % 0. 98
KNN 80. 4 % 0. 92 94. 5 % 0. 99
```
```
Table 2. PERFORMANCE EVALUATION OF PROPOSED METHOD ON DATASETS USING DIFFERENT CLASSIFIERS.
.
```
## Performance Evaluation


**21**

```
❑ Cross-Corpora Evaluation
✓ To test the generalizability of our method, we designed a two-stage cross-
corpora evaluation experiment using the proposed features.
✓ In the first stage of this experiment, weused WLDR dataset to train our model and
evaluated it on the FF++ dataset. We achieved 98. 8 % training accuracy but
attain a less accurate result of 49. 6 % and an AUC of 0. 69 on the test set.
✓ In the second stage of this experiment, we used the FF++ dataset for training
and the WLDR dataset for testing and achieved an accuracy of 93. 4 % on
training but attained less accurate results of 50 % accuracy and 0. 50 AUC on the
test set.
✓ Cross-corpora experiments attained lower results because both FF++ and WLDR
dataset are different from each other. For instance, videos are diverse in terms of
illumination and occlusion conditions, presence of accessories like glasses on the
face, and loss of details due to compressed video resolution.
```
## Performance Evaluation


**22**

```
❑ Performance comparison with contemporary methods.
Table 3. PERFORMANCE EVALUATION OF PROPOSED METHOD WITH CONTEMPORARY METHODS.
```
```
AUC
```
```
Methods Technique Results
Proposed
Method
```
```
MDHFD + SVM (FF++)
MDHFD + SVM (WLDR)
```
```
0.99
1.00
[ 9 ] SVM Classifier+ landmarks 0.89
[10] SVM+ Multimedia stream descriptor 0.93
[12] SVM classifier 0.96
[15] texture energy-based features + (MLP,
LogReg)
```
```
0.851
```
0. 784
[16] Landmarks+(VGG16,
ResNet50,
ResNet101,
ResNet152)

```
0.84
```
0. 97
0. 95
0.93

```
Accuracy
```
```
Proposed
Method
```
```
MDHFD + SVM (FF++)
MDHFD + SVM (WLDR)
```
```
92 .3%
96.9%
[8] SURF + SVM 92%
[13] EAR+ landmarks 87.5%
[21] Deep Features + CNN 83.71%
[22] Deep Features + SVM 90.29%
```
## Performance Evaluation


# Conclusion

**23**

```
 In this research work, we have presented a novel texture feature descriptor to
reliably capture the orientation and magnitude-oriented details from the input
video frames that are then used to detect the face swap deepfakes.
 The proposed method better addresses the problem of face swap-based
deepfakes detection under challenging conditions such as variations in the face skin
tone of people having different races, illumination conditions, presence of
accessories like glasses on the face, etc.
 We evaluated the performance of the proposed method on the face-swap subset
of Face Forensics++ and the WLDR dataset that contains all the aforementioned
challenges.
 Experimental results on both datasets illustrate the effectiveness of the proposed
method over the state-of-the-art methods for better detection of face-swap
deepfakes.
 In the future, we plan to extend our work to detect multiple types of deepfakes
and will try to improve our results on cross-corpora evaluation.
```

**24**
[1] FakeApp 2.2.0, Available at: https://www.malavida.com/en/soft/fakeapp/. Accessed: September 18, 2020.
[2] Faceswap: Deepfakes software for all, Available at: https://github.com/deepfakes/faceswap. Accessed: September 08, 2020.
[3] G. Antipov, M. Baccouche, and J.-L. Dugelay. Face aging with conditional generative adversarial networks. arXiv:1702.01983, Feb. 2017.
[4] A. Tewari et al. Mofa: Model-based deep convolutional face autoencoder for unsupervised monocular reconstruction. Proceedings of the IEEE International Conference on
Computer Vision Workshops, pages 1274[5] J. F. Boylan, "Will deep-fake technology destroy democracy?" The New York Times, Oct, vol. 17, 2018–1283, Oct. 2017. Venice, Italy. [
[6] A. Khodabakhsh, R. Ramachandra, K. Raja, P. Wasnik, C. Busch, in 2018 International Conference of the Biometrics Special Interest Group (BIOSIG). Fake face detection
methods: can they be generalized? (IEEE, 2018). https://doi.org/10.23919/biosig.2018.8553251.
[7] Y. Zhang, L. Zheng, and V. L. Thing, "Automated face swapping and its detection," in 2017 IEEE 2nd International Conference on Signal and Image Processing (ICSIP),
2017, pp. 15[8] A. Agarwal, R. Singh, M. Vatsa, A. Noore, in 2017 IEEE International Joint Conference on Biometrics (IJCB). Swapped! Digi-19: IEEE. tal face presentation attack detection via
weighted local magnitude pattern (IEEE, 2017).
[9] X. Yang, Y. Li, and S. Lyu, "Exposing deep fakes using inconsistent head poses," in ICASSP 2019- 2019 IEEE International Conference on Acoustics, Speech, and Signal
Processing (ICASSP), 2019, pp. 8261-8265: IEEE.
[10] D. Güera, S. Baireddy, P. Bestagini, S. Tubaro, and E. J. Delp, "We Need No Pixels: Video Manipulation Detection Using Stream Descriptors," arXiv preprint
arXiv:1906.08743, 2019. [11] K. Jack, "Chapter 13-MPEG-2," Video Demystified: A Handbook for the Digital Engineer, pp. 577- 737.
[12] Shruti Agarwal, Hany Farid, Yuming Gu, Mingming He, Koki Nagano, and Hao Li. Protecting world leaders against deep fakes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops, pages 38–45, 2019.
[13] T. Jung, S. Kim, and K. Kim, "DeepVision: Deepfakes Detection Using Human Eye Blinking Pattern," IEEE Access, vol. 8, pp. 83144- 83154, 2020.
[14] T. [15] F. SoukupovaMatern, C. and J. Cech, "Eye blink detection using facial landmarks," in 21st computer vision winter workshop, Riess, and M. Stamminger, "Exploiting visual artifacts to expose deepfakes and face manipulations," in 2019 IEEE Winter Applications of Computer VisiRimske Toplice, Slovenia, 2016. on
Workshops (WACVW), 2019, pp. 83-92: IEEE.
[16] Qurat-ul-ain, N. Nida, A. Irtaza, and N. Ilyas, "Forged Face Detection using ELA and Deep Learning Techniques," 2021 International Bhurban Conference on Applied
Sciences and Technologies (IBCAST), 2021, pp. 271-275, doi: 10.1109/IBCAST51254.2021.9393234.
[17] Akhtar, Z., & Dasgupta, D. (2019, November). A comparative evaluation of local feature descriptors for deepfakes detectiTechnologies for Homeland Security (HST) (pp. 1-5). IEEE. on. In 2019 IEEE International Symposium on
[18] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, "Joint face detection and alignment using multitask cascaded convolutional networks," IEEE Signal Processing Letters, vol. 23, no.
10, pp. 1499-1503, 2016.
[19] P. Korshunov, S. Marcel, Deepfakes: a new threat to face recognition? assessment and detection. arXiv preprint arXiv:1812.08685 (2018).
[20] D. Forensics and Security (WIFS), 2018, pp. 1Afchar, V. Nozick, J. Yamagishi, and I. Echizen, "-7: IEEE. Mesonet: a compact facial video forgery detection network," in 2018 IEEE International Workshop on Information
[21] H. H. Nguyen, F. Fang, J. Yamagishi, and I. Echizen, "Multi-task learning for detecting and segmenting manipulated facial images and videos," in 2019 IEEE 10th
International Conference on Biometrics Theory, Applications and Systems (BTAS), 2019, pp. 1-8.
[22] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, "Faceforensics++: Learning to detect manipulated facial images," in Proceedings of the IEEE
International Conference on Computer Vision, 2019, pp. 1[23] Masood, M., Nawaz, M., Malik, K. M., Javed, A., Irtaza, A., & Malik, H. (2022). Deepfakes Generation and Detection: Stat-11. e-of-the-art, open challenges,
countermeasures, and way forward. Applied Intelligence, 1- 53

# References


# For any query Feel Free to ask

**25**


