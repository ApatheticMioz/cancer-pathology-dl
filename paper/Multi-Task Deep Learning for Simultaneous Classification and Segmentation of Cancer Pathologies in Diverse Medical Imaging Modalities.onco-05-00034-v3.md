**==> picture [78 x 35] intentionally omitted <==**

## _Article_ 

## **Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities** 

**Maryem Rhanoui[1,] * , Khaoula Alaoui Belghiti[2] and Mounia Mikram[2,] *** 

- 1 Laboratory Health Systemic Process (P2S), UR4129, University Claude Bernard Lyon 1, University of Lyon, 69008 Lyon, France 

- 2 Meridian Team, LyRICA Laboratory, School of Information Sciences, Rabat 10100, Morocco; khaoula.alaoui-belghiti@esi.ac.ma 

- Correspondence: maryem.rhanoui@univ-lyon1.fr (M.R.); mmikram@esi.ac.ma (M.M.) 

## **Simple Summary** 

Academic Editor: Lorenzo Lo Muzio 

Received: 4 May 2025 Revised: 28 June 2025 Accepted: 6 July 2025 Published: 11 July 2025 

**Citation:** Rhanoui, M.; Alaoui Belghiti, K.; Mikram, M. Multi-Task Deep Learning for Simultaneous Classification and Segmentation of Cancer Pathologies in Diverse Medical Imaging Modalities. _Onco_ **2025** , _5_ , 34. https://doi.org/10.3390/ onco5030034 

**Correction Statement:** This article has been republished with a minor change. The change does not affect the scientific content of the article and further details are available within the backmatter of the website version of this article. 

**Copyright:** © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/). 

This research addresses the critical challenge of accurate cancer diagnosis from medical imaging when dealing with limited labeled data. Current deep learning approaches for cancer detection still face difficulties with insufficient training data and demanding accuracy requirements. We developed an automated multi-task deep learning system that simultaneously performs cancer classification and tumor segmentation across four major cancer types: skin lesions, brain tumors, prostate cancer, and pneumothorax. Using a UNet architecture with pre-trained backbones (VGG16 and MobileNetV2), the approach efficiently handles diverse medical imaging modalities including MRI, X-ray, dermoscopic, and histopathology images. The multi-task framework demonstrates superior performance compared to single-task approaches, achieving classification accuracies of 86–90% and segmentation precisions of 95–99% across different cancer types. These findings demonstrate that multi-task learning can effectively overcome data scarcity limitations while improving diagnostic accuracy, potentially accelerating AI-assisted cancer diagnosis deployment in clinical settings with limited resources. 

## **Abstract** 

_Background_ : Clinical imaging is an important part of health care providing physicians with great assistance in patients treatment. In fact, segmentation and grading of tumors can help doctors assess the severity of the cancer at an early stage and increase the chances of cure. Despite that Deep Learning for cancer diagnosis has achieved clinically acceptable accuracy, there still remains challenging tasks, especially in the context of insufficient labeled data and the subsequent need for expensive computational ressources. _Objective_ : This paper presents a lightweight classification and segmentation deep learning model to assist in the identification of cancerous tumors with high accuracy despite the scarcity of medical data. _Methods_ : We propose a multi-task architecture for classification and segmentation of cancerous tumors in the Brain, Skin, Prostate and lungs. The model is based on the UNet architecture with different pre-trained deep learning models (VGG 16 and MobileNetv2) as a backbone. The multi-task model is validated on relatively small datasets (slightly exceed 1200 images) that are diverse in terms of modalities (IRM, X-Ray, Dermoscopic and Digital Histopathology), number of classes, shapes, and sizes of cancer pathologies using the accuracy and dice coefficient as statistical metrics. _Results_ : Experiments show that the multi-task approach improve the learning efficiency and the prediction accuracy for the segmentation and classification tasks, compared to training the individual models 

https://doi.org/10.3390/onco5030034 

_Onco_ **2025** , _5_ , 34 

_Onco_ **2025** , _5_ , 34 

2 of 18 

separately. The multi-task architecture reached a classification accuracy of 86%, 90%, 88%, and 87% respectively for Skin Lesion, Brain Tumor, Prostate Cancer and Pneumothorax. For the segmentation tasks we were able to achieve high precisions respectively 95%, 98% for the Skin Lesion and Brain Tumor segmentation and a 99% precise segmentation for both Prostate cancer and Pneumothorax. Proving that the multi-task solution is more efficient than single-task networks. 

**Keywords:** segmentation; classification; multi-task learning; deep learning; oncology 

## **1. Introduction** 

Cancer is one of the leading causes of death in the world [1]. Every year, tens of millions of people are diagnosed with cancer and more than half of these patients die. In most cases, cancers are detected at an advanced stage, making treatment more difficult and very costly. Decades of research have focused on obtaining an accurate early diagnosis of cancer so that doctors can find the best treatment at the right time. When cancer care is delayed or inaccessible, the chances of survival are lower, the problems associated with treatment are greater, and the cost of care becomes higher. 

Early diagnosis is the key to effective treatment and can improve cancer outcomes by providing care in the earliest possible stage representing an important public health strategy [2]. However, the cancer diagnosis process is complicated and requires a lot of time and human effort. Therefore, it is very important to provide efficient analysis tools that can extract relevant information from medical images to speed up diagnosis and make it safer and more reliable. 

In such cases, artificial intelligence (AI) is iterative to help optimize the diagnosis process [3–5], especially recent advances in deep learning, as we have recently noticed that AI is being thoroughly considered and widely applied to all aspects of medical image analysis [6–8], including oncology [9–12], making a high-precision diagnosis possible. It has proven to be one of the best ways to analyze and make predictions based on large imaging data sets. The analysis of medical images is a critical task that needs to address several challenges: first, in the medical field in general and oncology in particular, we need to have the best accuracy, since it involves human lives [13]. Secondly, the images are of different modalities (MRI, X-ray, etc.) and can contain several classes depending on the nature of the pathologies treated. And finally, the size of the data sets is relatively small and insufficient to efficiently train resource-intensive deep learning models [14], usually addressed by generating synthetic images [15], which is not optimal in a context of inter-modality. 

Through medical image analysis, classification and segmentation tasks are extremely important and provide a better understanding of cancer and a better definition of the most effective treatment [16]. Both techniques are useful in extracting features, analyzing, and interpreting medical images. These techniques are used to find organs, tumors and other anatomical structures, with the aim of acquiring quantitative information useful for decision making in diagnosis, surgery, or clinical studies. Many of the indicators used to assess cancer risk and severity are derived from the tumor segmentation and its grading. 

Deep learning techniques and specifically convolutional neural networks (CNN) models are extremely promising in such cases. These models are bringing great success in many computer vision applications such as medical image classification [17] and segmentation [18], unfortunately most of the proposed deep learning approaches handle these two 

_Onco_ **2025** , _5_ , 34 

3 of 18 

tasks independently despite their interconnectedness which requires much more effort, computational resources and time to execute. 

Multi-task learning MTL [19] is a more optimal and best performing solution for this kind of resources problems related to Deep Learning models. It consists in introducing a related auxiliary task in the network to learn both tasks simultaneously. This auxiliary task not only brings additional data but also helps the network to generalize better and learn a more powerful representation from the data. The idea of multi-task learning is inspired by the way humans use the knowledge gained in other related tasks, in learning the target task, in order to increase its performance. 

Multi-task learning enables simultaneous training of classification and segmentation tasks within a single model by sharing internal feature representations across network layers. This approach enhances computational efficiency, reduces resource consumption, and improves predictive accuracy. It has demonstrated strong performance across various medical domains, including neurodegenerative disease analysis [20,21], oncology [22–24], and COVID-19 diagnostics [25–27]. It is also increasingly used for unstructured medical texts [28–30], enabling models to simultaneously extract clinical entities, classify outcomes, and detect relationships within reports or notes. 

However, the proposed solutions are mainly mono-task based models, and they suffer from the issue of data sparsity which can negatively impact the performances. Moreover, the proposed multi-task models, although they benefit from the advantages of MTL, they have been mostly tested and validated on a single dataset, which does not help to assert the generalization and application of the approach on the different medical images in oncology. In this context, the objective of this paper is to implement a deep learning-based solution for the classification and segmentation of different types of cancer from medical images with several modalities. 

To meet this objective, we built two multi-task CNN models processing the classification and segmentation tasks. Both models are based on the UNet architecture [31] by replacing its encoder with pretrained models (VGG16, Mobilenetv2). Multi-task models were trained and evaluated on four different datasets, and their performance was compared to a single task model of image segmentation and classification namely Attention based UNet, Mask-RCNN for segmentation and VGG16, Mobilenetv2 for classification. The obtained results show very encouraging performances of the multi-task solution compared to the single-task models. 

The reminder of this paper is organized as follows: In Section 2, we present the technical background, then we summarize in Section 3 the related works. Section 4 presents our proposed approach. In Section 5 the experimental setup and finally the last section discusses the results. 

## **2. Background** 

## _2.1. Transfer Learning with Convolutional Neural Networks_ 

Convolutional Neural Networks (CNN) is gradually being applied to image classification and segmentation as deep learning gains traction, greatly boosting the accuracy of both tasks. A convolutional neural network is a type of acyclic (feed-forward) artificial neural network, in which the connection pattern between neurons is inspired by the visual cortex of animals. The neurons in this brain region are arranged so that they correspond to overlapping regions when paving the visual field; as the first layers in CNN learn filters for detecting basic features: edges, corners, etc. These first layers features are data or task independent, but are general to various data and tasks. 

Transfer learning involves training a network on a base dataset and task, and then reassigning or transferring the learned features to a target network that will be trained on 

_Onco_ **2025** , _5_ , 34 

4 of 18 

a target dataset and task. The extracted features are general and independent of the base and target tasks, allowing us to: Learn robust features and reduce the number of trainable parameters, Help the model converge much faster, and achieve higher performance compared to one trained from scratch. 

## 2.1.1. VGG16 

The pretrained VGG16 convolutional neural network model [32] used to win ILSVR competition on ImageNet dataset, which contains over a million images and can classify images into 1000 object categories [33]. Due to its simplicity and high performance, VGG16 model was considered as one of the excellent vision model architectures. 

Based on sixteen weighted layers. VGG16 takes as a default input size 224 _×_ 224 pixels with 3 RGB channels images. Its architecture consists of a series of convolution layers with 3 _×_ 3 filter stride 1 and maxpool layers of 2 _×_ 2 filter of stride 2. 

## 2.1.2. MobileNetV2 

MobileNet, mainly a lightweight deep neural network proposed by Google based on the decomposition of convolution kernel, It can effectively reduce network parameters while taking into account optimization delay, excellent robustness while keeping the number of model parameters effectively reduced. 

The second version of MobileNet (MobileNetV2) [34] is a simplified architecture that uses depth-separable convolutions to build deep and lightweight convolutional neural networks and provides an efficient model for classification applications; using lightweight depthwise convolutions to filter features in the intermediate expansion layer. 

It has also improved the state of the art performance of mobile models on multiple tasks such as classification and object detection and benchmarks as well as across a spectrum of different model sizes. 

## _2.2. Multi Task Learning_ 

Multi-task Learning (MTL) [19,35] is a learning paradigm in the field of machine learning and its goal is to take advantage of the useful information contained in several related tasks in order to improve the generalization performance of all the tasks. MTL can learn robust representations between related tasks. These shared representations increase the efficiency of the data which leads to better performance and a mitigation of the risk of excessive overfitting. 

In the problem of data scarcity, the number of labeled data in each task is insufficient to train an accurate model, while MTL uses more of the data from different tasks to get a more accurate result for each task, also having a single shared model instead of independent models per task reduces storage space and alleviates well-known weaknesses in deep learning: data requirements and compute demand. 

Multi-task learning is commonly accomplished in deep learning using either hard or soft parameter sharing of hidden layers. Hard parameter sharing is the most commonly used approach for MTL with neural networks. It is usually applied by sharing the hidden layers among all tasks, while keeping several task-specific output layers, which greatly reduces the risk of overfitting. In Soft parameter sharing type, each task has its own model with its own parameters, the distance between the model parameters is then regularized to encourage the parameters to be similar. 

## **3. Related Works** 

Several architectures based on deep learning have been proposed for the classification and segmentation of medical images. 

_Onco_ **2025** , _5_ , 34 

5 of 18 

In [25] Amyar et al. presented a multi-task deep learning model to jointly identify whether the patient has COVID-19 and segment the COVID-19 lesion from the CT images. The architecture is composed of an encoder and two decoders for reconstruction and segmentation, and a multilayer perceptron for classification. Also in [36], Aram et al. presented a model that combines instance segmentation, the LSTM long-term memory network, and the attention mechanism to predict COVID-19 and segment CT images. The model works by extracting a sequence of regions of interest that contain relevant class information (COVID-19, Common Pneumonia, Control) and applies two LSTMs with an attention mechanism to this sequence to extract relevant features for determining class. 

Thi et al. [37] proposed a multi-task learning scheme, which combines segmentation and classification for cancer diagnosis in mammography. The proposed architecture is based on a fully convolutional network (FCN) that allows efficient sharing of features. They concluded that joint training allows for better cooperation between tasks. 

For skin lesion analysis [38] Yang et al. proposed a multi-task deep neural network. The proposed multi-task learning model solves the tasks of lesion segmentation and lesion classification. 

Gu et al. [39] proposed an attention-based CNN (CA-Net). The attention mechanism is useful for improving the segmentation performance of CNNs because it focuses on the most relevant information in the feature maps while removing irrelevant parts. CA-Net significantly improved the mean Dice segmentation score from 87.77% to 92.08% for the skin lesion, from 84.79% to 87.08% for the placenta, and from 93.20% to 95.88% for the fetal brain, respectively, compared to UNet. 

Jain et al. [40] developed an efficient multi-task model for simultaneous segmentation and classification for dilated cardiomyopathy, using an encoder/decoder architecture with the attention mechanism. This model is based on an encoder/decoder architecture and consists of three branches: The extraction path (encoder), The attention path which has the advantage of being able to highlight features that are advantageous for classification tasks and finally the recovery path, which allows the model to learn higher level intermediate representations. The experimental results show that the multi-task network achieved an accuracy of 97.63%, an AUC of 98.32%. By comparing their model with other classification and segmentation techniques they showed that the performance of multi-task segmentation and classification is better than that of single-task segmentation or classification. 

Foo et al. [41] proposed a multi-task deep learning system called MTUnet, for lesion segmentation and classification of Diabetic retinopathy DR, the image of the diabetic retina can be classified into different severity levels 0 (no apparent DR), 1 (mild DR), 2 (moderate DR), 3 (severe DR), 4 (poliferative DR). This system is an encoder-decoder network based on Unet architecture by replacing its encoder with a classical VGG16 network to produce the DR classification and the decoder for lesion segmentation. 

Zhou el al. [24] aims to overcome the shortcomings of the popular cascading model strategy that can lead to undesirable system complexity due to its multiple separate models. As a solution, they proposed to adopt multi-task learning to integrate multiple tasks into a single run (OM-Net). They decomposed the multi-class segmentation of Brain Tumors into three distinct but interconnected tasks: (1) Segmentation to detect complete tumor, (2) Refined segmentation for the complete tumor and its intra-tumor classes and (3) Precise segmentation. Each task has its specific parameters and the shared backbone model aims to learn the correlation between tasks. The guided attention module (CGA) is added to share prediction results between OM-Net tasks, which can use the prediction results generated by the previous task to produce more refined category-specific statistics. 

To preserve contextual information, Gu et al. [42] suggested utilizing a dilated convolution block at the network’s bottleneck. In this work the authors present a contextual 

_Onco_ **2025** , _5_ , 34 

6 of 18 

encoding network (called CE-Net) to capture more high-level information and preserve image integrity. This CE-Net is established for medical image segmentation and compared with UNet. The proposed CE-Net adopts a pretrained ResNet block in the encoder part. A new “Context extractor” is added to extract context-related semantic information. They applied the proposed CE-Net to different 2D medical image segmentation tasks. The results show that the proposed method exceeds the performance of the original UNet method and other methods for “Optic disc” segmentation, lung segmentation, cell contour segmentation. 

Lv et al. [43] introduced BrainTumNet, a transformer-enhanced multi-task deep learning model utilizing an advanced encoder–decoder architecture with adaptive masked Transformers and multi-scale feature fusion. This network achieved optimale performance, with an IoU of 0.921 and DSC of 0.91 for tumor segmentation, alongside a classification accuracy of 93.4% evaluated across both internal and external datasets. 

## **4. Multi Task Unet Model Description** 

## _4.1. Multi-Task Model Architecture_ 

We use the multi-task architecture that combines transfer and multi-task learning [41], where the coding block is tested by the convolutional layers of VGG16 (Figure 1) and MobileNetv2 (Figure 2) networks, which are pretrained on the ImageNet dataset. The use of transfer learning allows robust features to be learned and reduces the number of trainable parameters. It also helps the model converge much faster compared to a model that is not pretrained. A pretrained model achieves high performance compared to a model trained from scratch. 

**==> picture [407 x 249] intentionally omitted <==**

**Figure 1.** Multi-task UNet Architecture with VGG16 Encoder. 

The multi-task network model allows the different components to share the features detected in the first layers. The network is composed of different components that will be trained using an annotated dataset. The network is composed of four blocks for an analysis of the data set on cancer pathologies: 

_Onco_ **2025** , _5_ , 34 

7 of 18 

- Feature Extraction. • Classification Task. 

- • Bottleneck Block. • Segmentation Task. 

**==> picture [364 x 116] intentionally omitted <==**

**Figure 2.** Multi-task UNet Architecture with MobileNetV2 Encoder. 

The detected features learned by the multi-task model can be used for the prediction of the two tasks, i.e., the classification and segmentation of the cancer cells in the dataset. The multi-task approach is an encoder-decoder network architecture and is depicted schematically in Figure 1. 

The encoder network is used to capture the context of the image. It is a traditional stack of convolution and Maxpool layers. During these convolution and subsampling operations the spatial information is lost, which allows the network to learn “What” is in the image, but at the same time it loses the “Where” information, the encoder is similar to a traditional classification network; it follows the typical architecture of a CNN with the repeated application of convolutional and subsampling layers. With this multi-task architecture we will take advantage of this path to perform the classification task instead of training an independent CNN model, so we will use the encoder as a base and we will add “Fully connected layers” that will produce a label corresponding to the input image. 

The Bottleneck is the layer containing a compressed representation of input data. At this level, there are two steps to do: classification and segmentation. The first task produces the classification label using the encoder path, while the second produces the segmentation mask following the decoder path. After going through the encoder path, we get a coded version of the input image. 

In semantic segmentation, the information about the spatial dimensions of the image must be preserved, so the decoder will build the output from the compressed representation, this decoder is a stack of oversampling and convolution blocks which helps to recover the lost spatial information, and to do this it uses “Skip connections” between the encoder and the decoder, at each stage of the decoder, concatenating the output of the oversampling layers with the corresponding feature maps of the encoder at the same level. These connections are useful for transferring information from the low-level layers of the encoder path to the high-level layers of the decoder path, as this information is needed to generate reconstructions that have precise details. 

## _4.2. Optimization and Evaluation Metrics_ 

Multi-task learning concerns the problem of multi-objective optimization of a model. In this paper, the multi-task network uses two loss functions from the segmentation and the classification tasks. More specifically, the segmentation task’s loss function affects the entire network (encoder and decoder), whereas the classification task uses a weighted categorical cross-entropy loss that only affects the encoder section. The classification loss incorporates sklearn’s balanced class weighting strategy, where weights are calculated using the inverse class frequency with the ’balanced’ method for each of the diagnostic categories, helping to 

_Onco_ **2025** , _5_ , 34 

8 of 18 

overcome the challenge of class imbalance in the datasets used. The overall network loss function is a linear combination of the losses for each individual task. 

The first task is a multi-class classification problem, the prediction outputs a probability vector _P_ consisting of the predicted probabilities of all classes. Then, _P_ is compared to the grading ground-truth _G_ . The categorical cross-entropy loss is used, as shown in Equation: 

**==> picture [162 x 28] intentionally omitted <==**

where _N_ is the number of examples in the training set, _C_ is the number of categories. The output _Pic_ may be regarded as the predicted probability distribution for the _i_ th observation belonging to class _c_ , and the target _Gic_ can be interpreted as true. 

The segmentation task outputs a prediction map where each pixel contains a class label represented as an integer. Since it is a binary classification of whether the pixel is cancer or background, the loss function used is binary cross entropy computing by the following average: 

**==> picture [231 x 26] intentionally omitted <==**

The total loss is the average loss across all the examples, where _n_ is the number of ˆ examples, _y_ represents the true label, _yi_ denotes a single element of that label, _y_ represents ˆ the predicted value by the model and _yi_ refers to a single element of that prediction. 

To enable joint optimization of both classification and segmentation tasks, we define the total loss _L_ total as a weighted sum of the individual task-specific losses: 

**==> picture [133 x 12] intentionally omitted <==**

where _L_ seg denotes the segmentation loss, and _L_ cls represents the classification loss. The hyperparameters _λ_ seg and _λ_ cls control the relative contribution of each task to the total loss. 

In our experiments, we empirically selected a weighting of _λ_ seg = 5 and _λ_ cls = 1, which provided the best trade-off between segmentation precision and classification performance (see Section 5.3). This reflects the increased complexity and spatial nature of the segmentation task, which benefits from a stronger optimization signal. 

To evaluate and compare the performance of the classification and segmentation tasks of the UNet multi-task model, we use two metrics: 

## 4.2.1. Accuracy 

Measuring the ratio between the correct predictions of the classification model and the total number of predictions. 

**==> picture [149 x 22] intentionally omitted <==**

## 4.2.2. Dice Similarity Coefficient 

The most widely used metric for evaluating segmentation performance; characterizing the agreement between the segmented pixels and the ground truth. The coefficient ranges from 0 (no overlap) to 1 (complete overlap). Formulated as follows: 

**==> picture [99 x 25] intentionally omitted <==**

where _G_ is the ground truth, and S the segmentation pixels, _G ∩ S_ represents the common elements between the sets _G_ and _S_ , and |.| denotes the set cardinality. 

_Onco_ **2025** , _5_ , 34 

9 of 18 

## **5. Experiments** 

We conduct a comparative study to assess multi-task UNet models performance for both tasks on various datasets with several medical imaging modalities including IRM, X-Ray, dermoscopic and digital histopathology. The multi-task UNet model is used with different pre-trained deep learning models (VGG 16 and MobileNetv2). 

## _5.1. Data Description_ 

We propose a multi-task learning approach for the joint classification and segmentation of cancer-related medical images using a unified model. This framework requires datasets annotated for both tasks, necessitating additional preprocessing efforts to generate consistent labels and segmentation masks. 

To evaluate the generalizability of the model, we curated and processed heterogeneous datasets varying in format, size, and origin, each corresponding to a distinct cancer type. Experiments were conducted on datasets covering skin lesions, brain tumors, and prostate cancer. Additionally, a dataset on pneumothorax was included to assess the applicability of the multi-task approach to non-cancerous pathologies, thereby highlighting the model’s versatility. 

## 5.1.1. Skin Lesion: ISIC 2018 Dataset 

The ISIC 2018 dataset [44] was released by the International Skin Imaging Collaboration (ISIC) as a large-scale dataset of dermoscopic images. Skin lesions are characterized by distinct spots, abnormal bumps, scars, blisters, or different discoloration of the skin on normal body areas. This dataset contains 2594 RGB images of skin lesions, each with a resolution of 2166 _×_ 3188 pixels. Figure 3 shows sample skin lesions from the ISIC 2018 dataset. 

**==> picture [295 x 62] intentionally omitted <==**

**Figure 3.** Samples of Skin Lesions from ISIC 2018 dataset. 

The lesions are classified into the following labels: actinic Keratoses, basal cell carcinoma, benign keratosis, dermatofibroma, melanocytic nevi, vascular skin lesions and melanoma. 

## 5.1.2. Brain Tumor: TCGA-LGG Dataset 

Of all brain tumors, glioma is the most serious. According to World Health Organization (WHO) criteria, gliomas have been classified into four grades, from grade I to grade IV, based on the malignancy of the tumor. Normally, grade III and IV gliomas are called high-grade gliomas (HGG) and grades I and II are called low-grade gliomas (LGG). LGGs can be classified into astrocytomas, oligodendrogliomas and oligodendrocyte astrocytomas according to the pathological type. In our work we will use the LGG type gliomas. 

The Cancer Genome Atlas Low Grade Glioma (TCGA-LGG) dataset will be used as part of a larger effort to create a research community focused on linking cancer phenotypes to genotypes by providing clinical images matched to Cancer Genome Atlas (TCGA) subjects (Figure 4). 

_Onco_ **2025** , _5_ , 34 

10 of 18 

**==> picture [295 x 73] intentionally omitted <==**

**Figure 4.** Samples of Brain Tumor from TCGA-LGG dataset. 

## 5.1.3. Prostate Cancer: PANDA Prostate Cancer Grade Assessment Dataset 

The data used in this study is part of a Kaggle competition to detect prostate cancer [45]. The Diagnosis is based on the grading of prostate tissue biopsies. These tissue samples are examined by a pathologist and scored according to the Gleason grading system. In this dataset the training set consists of about 11.000 prostate biopsies, with slide level labels and label masks. The Karolinska Institute and Radboud University Medical Center, respectively, gathered and labeled the samples. 

The grading process consists of finding and classifying cancer tissue into Gleason patterns based on the tumor’s architectural growth patterns. After marking a biopsy report into the corresponding Gleason score, it is converted into an ISUP grade on a 1–5 scale. Figure 5 shows sample prostate cancer images from the PANDA dataset. 

**==> picture [305 x 59] intentionally omitted <==**

**Figure 5.** Samples of Prostate Cancer from PANDA Dataset. 

## 5.1.4. SIIM-ACR Pneumothorax Segmentation Dataset 

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect and segment pneumothorax would be useful in many clinical scenarios. 

The Society for Imaging Informatics in Medicine (SIIM) has organized a competition to assist in the detection of pneumothorax from radiological images. The Society is a leading healthcare organization for those interested in the current and future use of informatics in medical imaging. Its mission is to advance medical imaging informatics across the enterprise through education, research and innovation in a multidisciplinary community. 

Figure 6 shows sample images from the SIIM-ACR Pneumothorax Segmentation dataset. 

**==> picture [294 x 76] intentionally omitted <==**

**Figure 6.** Samples from SIIM-ACR Pneumothorax Segmentation dataset. 

A general overview of the datasets is explained in Table 1 and the distribution of the classes in the datasets is shown in Table 2. 

_Onco_ **2025** , _5_ , 34 

11 of 18 

**Table 1.** Publicly Available Medical Image Datasets used in our tests. 

|**Dataset**|**Organ**|**Number of Images**|**Input Size**|**Image Type**|**Classes**|
|---|---|---|---|---|---|
|ISIC 2018|Skin Lesion|10,015|(224, 224)|Dermoscopic image|7|
|TCGA-LGG|Brain Tumor|3929|(256, 256)|IRM|2|
|Prostate Cancer<br>Grade Assessment|Prostate Cancer|10,616|(128, 128)|Digital Histopathology|6|
|SIIM-ACR Pneumothorax<br>Segmentation|Pneumothorax|12,047|(224, 224)|X-Ray|2|



**Table 2.** Distribution of classes in datasets. 

|**Dataset**|**Class**|**Train**|**Test**|**Total**|
|---|---|---|---|---|
||0 Actinic keratoses|262|65|327|
||1 Basal cell<br>carcinoma|412|102|514|
|Skin Lesion|2 Benign keratosis|879|220|1099|
||3 Dermatofbroma|92|23|115|
||4 Melanocytic nevi|891|222|1113|
||5 Vascular skin lesi|5341|1341|6705|
||6 Melanoma|114|28|142|
|Brain Tumor|0 No tumor<br>1 Tumor|2173<br>1168|383<br>205|2556<br>1373|
||0 Background|2314|578|2892|
||1 Stroma|2133|533|2666|
|Prostate Cancer|2 Benign<br>epithelium|1074|269|1343|
||3 Gleason 3|996|248|1244|
||4 Gleason 4|1000|249|1249|
||5 Gleason 5|979|245|1224|
|Pneumothorax|0 No pneumoth<br>1 Pneumothorax|7503<br>2136|1875<br>533|9378<br>2669|



## _5.2. Preprocessing and Environment_ 

For efficient learning, deep artificial neural networks need a large corpus of training data. Collecting training data is often expensive and time-consuming. Data augmentation (DA) overcomes this problem to better serve learning by artificially extending the training set with several transformations, DA is a strategy that allows practitioners to significantly increase the diversity of data available for model training without collecting new data, Therefore, in this work, in order to expand the training datasets; the rotation, flipping and shearing modifications were used to augment the samples during our training, as shown in Figure 7. 

Regarding the environment used and the implementation of the approach; we utilized T4 GPUs from Google Colab’s cloud service environment to train our multi-task UNet architecture for classification and segmentation. The model was trained over 50 epochs with a batch size of 32, with an early stopping enabled. Also employed the Adam optimizer set to its default learning rate of 0.001. 

In the next section, we present the performance results for the classification and segmentation tasks, as well as the comparison with single-task approaches. 

_Onco_ **2025** , _5_ , 34 

12 of 18 

**==> picture [272 x 195] intentionally omitted <==**

**Figure 7.** Transformations applied on the original images. 

## _5.3. Ablation Study_ 

## 5.3.1. Skip Connections 

As described in the multi-task model architecture part, our model benefits from the learning gained with The Skip connections. between the decoder and encoder layers (see Figure 8), in order to ensure a constant information transfer helping to fine-gain more details used for the class prediction, which improves the model average loss in all four cases. 

**==> picture [376 x 72] intentionally omitted <==**

**Figure 8.** Multi-task UNet with MobileNetV2 Encoder: With and without Skip connections. 

To validate our hypothesis, we performed an ablation study comparing model architectures with and without skip connections. The experimental results demonstrate that skip connections consistently boosts performance in the four distinct disease cases. As evidenced by the performance metrics, this architectural enhancement not only improves model accuracy but also mitigates information loss; a key limitation observed in models lacking skip connections. 

## 5.3.2. Task Weight Allocation 

To investigate the impact of task prioritization in our multi-task setup, we ran a small experiment over 50 with a reduced set of data, varying the segmentation-to-classification loss weights as follows: [1:1], [2:1], [5:1], [10:1], [1:2], [1:5], and [1:10], while keeping all other settings constant. 

Notably, the best segmentation accuracy (0.9366) was obtained with a [5:1] weighting (Segmentation:Classification), which is the configuration used in our final proposed model. This setup allowed the segmentation branch to receive a stronger optimization signal while maintaining a reasonable classification accuracy (0.8021). 

Increasing segmentation weight beyond [5:1] (e.g., [10:1]) led to negligible segmentation improvement but worsened classification performance and increased the total loss. Similarly, increasing classification weight had a destabilizing effect on segmentation. This experiment highlight the importance of carefully tuning task weight allocation and validate 

_Onco_ **2025** , _5_ , 34 

13 of 18 

our design choice in giving higher weight to segmentation, which is the more complex and pixel-level task in our problem formulation. 

## **6. Results and Discussion** 

The specificity of medical data analysis implies the consideration of additional factors and must meet critical challenges to effectively achieve its main purpose. Despite having relatively small data sets, we were able to achieve good classification and segmentation results by exploiting the useful information contained in these two tasks for greater generalization with an effective image representation. This representation is the driving force behind the success of the multi-task model in outperforming single-task models. 

Table 3 summarizes the main results of the different experiments, we note that both multi-task models outperformed the single-task Attention-UNet model and Mask-RCNN for segmentation and VGG16/Mobilenetv2 for classification. For the classification task, the multi-task model MobilenetV2 outperforms the VGG16 one with 86%, 89% and 88% of accuracy in skin, brain and prostate respectively. Compared to the single-task equivalent models, the multi-task improves significantly the accuracy from 67% up to 90% for brain tumor classification, as well as from 74% to 86% for skin lesion classification and from 78% to 88% for prostate tumor classification. 

**Table 3.** Classification and Segmentation results: Performance comparison multi-task VS Monotask. 

|**Datasets**|**Mono-Task**<br>**Multi-Task**|
|---|---|
||**Classifcation**<br>**Segmentation**<br>**VGG16 Encoder**<br>**Mobilenetv2 Encoder**|
||**VGG-16**<br>**Mobilnetv2**<br>**Unet**<br>**Mask-RCNN**<br>**Classif**<br>**Segment**<br>**Classif**<br>**Segment**|
|Skin Lesion<br>Brain Tumor<br>Prostate Cancer<br>Pneumothorax|0.74<br>0.83<br>0.88<br>0.94<br>0.83<br>0.95<br>**0.86**<br>**0.95**<br>0.67<br>0.86<br>0.96<br>0.95<br>**0.90**<br>0.97<br>0.89<br>**0.98**<br>0.78<br>0.74<br>0.85<br>0.97<br>0.87<br>0.98<br>**0.88**<br>**0.99**<br>0.73<br>0.77<br>0.83<br>0.95<br>0.82<br>0.99<br>**0.87**<br>**0.99**|



Regarding the classification task, we used the Gradient-weighted Class Activation Mapping (Grad-CAM) [46] method as a visual tool expressing the model’s way of thinking in the process of classification, highlighting the regions indicating the probability that the image belongs to a predefined class. As the images in Figure 9 clearly illustrate, these attention maps reflects which parts of the brain, skin, prostate or lung is getting the model’s attention most to improve its classification process. 

For the segmentation task, the multi-task model produced the best image segmentation results for the three cancer datasets, with a dice coefficient of 95%, 98%, 99% respectively against 80%, 96%, 85% for the single-task Attention-UNet model, also outperforming the Mask-RCNN model with 94%, 95%, 97% respectively for the Skin, Brain and Prostate cancer’s segmentation.The best dice coefficients for the three datasets were obtained with the MobilnetV2-based multi task model. 

As shown above, the multi-task model improves the performance of the single-task models of cancer tumor image classification and segmentation. To further validate our model and its power of generalization, we apply it to a different objective which is the prediction of pneumothorax. As a result, the model also improved the performance of the classification and segmentation tasks of these images from 73%, and 77% with the classification single-task models to 87% and from 83%, 95% segmentation single-task models to 99% with the proposed solution. This confirms the good generalization performance of the multi-task model. 

As presented in Figure 10; the model is able to precisely segment the region of disease in the scan image, and present an empty scan when classified as non cancerous as in the case of prostate image. 

_Onco_ **2025** , _5_ , 34 

14 of 18 

**==> picture [286 x 299] intentionally omitted <==**

**Figure 9.** Grad-CAM on samples from the four Datasets. 

**==> picture [283 x 342] intentionally omitted <==**

**Figure 10.** Segmentation results: Predicted masks overlay. 

_Onco_ **2025** , _5_ , 34 

15 of 18 

Unlike previous studies, where the proposed multi-task models are usually validated on a single cancer type. Our solution proves effectiveness in both tasks applied on different cancer types and different data set modalities such as number of classes, image shapes and sizes of cancer pathologies. 

On another note, the statistical validation of our proposed method through 5-fold cross-validation demonstrates robust and significant improvements in segmentation performance. The paired _t_ -test analysis revealed a highly significant enhancement in Dice scores (0.995 ± 0.001 vs. 0.829 ± 0.034, _p_ < 0.001), representing a substantial 16.6 percentage point improvement over the baseline approach. This large effect size, combined with the consistent performance across all folds (Dice scores ranging from 0.994 to 0.996), indicates that our method not only achieves superior segmentation quality but does so reliably across different data partitions. While accuracy metrics remained comparable between methods (99.4%), the dramatic improvement in Dice scores suggests our approach excels particularly in precise boundary delineation and overlap accuracy, which are critical for clinical applications requiring high segmentation fidelity. 

## **7. Conclusions** 

In this study, we proposed an innovative multi-task deep learning framework for the simultaneous classification and segmentation of diverse cancer types from medical imaging data. The developed approach leverages the strength of multi-task learning, effectively enhancing model performance in scenarios characterized by limited labeled datasets and multiple imaging modalities, including MRI, X-ray, dermoscopic, and digital histopathology images. 

Our results demonstrate that the multi-task architecture consistently outperforms conventional single-task methods across various cancer datasets, achieving classification accuracies ranging from 86% to 90% and segmentation precisions between 95% and 99%. Notably, employing pre-trained models such as VGG16 and MobileNetV2 significantly improved the model’s feature extraction capabilities and reduced computational demands, thereby facilitating efficient and robust predictions. 

Moreover, the conducted ablation studies confirmed the effectiveness of critical design choices, such as the incorporation of skip connections and optimized task-specific loss weighting, which substantially contributed to performance improvements. 

This research underscores the potential of multi-task learning paradigms in overcoming data scarcity issues and enhancing diagnostic precision. By integrating classification and segmentation tasks into a single unified model, our solution offers a scalable and generalizable approach, applicable across diverse clinical scenarios and imaging modalities. 

Future work will explore further architectural enhancements, including attention mechanisms [47] and dilated convolutions [48], to refine spatial context modeling. Additionally, incorporating multimodal strategies [49] and multi-label classification capabilities [50] promises to capture more nuanced pathological information, ultimately supporting more comprehensive clinical decision-making. 

**Author Contributions:** Conceptualization, M.R. and M.M.; data curation, K.A.B.; writing, M.R. and M.M.; software, K.A.B. All authors have read and agreed to the published version of the manuscript. 

**Funding:** This research received no external funding. 

**Institutional Review Board Statement:** Not applicable. 

**Informed Consent Statement:** Not applicable. 

**Data Availability Statement:** All datasets used are publicly available: the ISIC 2018 dataset can be accessed via the ISIC Archive at https://challenge.isic-archive.com/data; the TCGA-LGG dataset is 

_Onco_ **2025** , _5_ , 34 

16 of 18 

available through the GDC Data Portal under project TCGA-LGG at https://portal.gdc.cancer.gov/; the PANDA prostate cancer dataset is hosted on Kaggle at https://www.kaggle.com/c/prostatecancer-grade-assessment; and the SIIM-ACR Pneumothorax Segmentation dataset is available on Kaggle at https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation, accessed on 28 June 2025. 

**Conflicts of Interest:** The authors declare no conflicts of interest. 

## **References** 

1. Siegel, R.L.; Miller, K.D.; Fuchs, H.E.; Jemal, A. Cancer statistics, 2021. _CA Cancer J. Clin._ **2021** , _71_ , 7–33. [CrossRef] 

2. Etzioni, R.; Urban, N.; Ramsey, S.; McIntosh, M.; Schwartz, S.; Reid, B.; Radich, J.; Anderson, G.; Hartwell, L. The case for early detection. _Nat. Rev. Cancer_ **2003** , _3_ , 243–252. [CrossRef] 

3. Elemento, O.; Leslie, C.; Lundin, J.; Tourassi, G. Artificial intelligence in cancer research, diagnosis and therapy. _Nat. Rev. Cancer_ **2021** , _21_ , 747–752. [CrossRef] [PubMed] 

4. Hunter, B.; Hindocha, S.; Lee, R.W. The role of artificial intelligence in early cancer diagnosis. _Cancers_ **2022** , _14_ , 1524. [CrossRef] [PubMed] 

5. Zekaoui, N.E.; Yousfi, S.; Mikram, M.; Rhanoui, M. Enhancing large language models’ utility for medical question-answering: A patient health question summarization approach. In Proceedings of the 2023 14th International Conference on Intelligent Systems: Theories and Applications (SITA), Casablanca, Morocco, 22–23 November 2023; pp. 1–8. 

6. Ting, D.S.; Liu, Y.; Burlina, P.; Xu, X.; Bressler, N.M.; Wong, T.Y. AI for medical imaging goes deep. _Nat. Med._ **2018** , _24_ , 539–540. [CrossRef] 

7. Giger, M.L. Machine learning in medical imaging. _J. Am. Coll. Radiol._ **2018** , _15_ , 512–520. [CrossRef] 

8. Currie, G.; Hawk, K.E.; Rohren, E.; Vial, A.; Klein, R. Machine learning and deep learning in medical imaging: Intelligent imaging. _J. Med Imaging Radiat. Sci._ **2019** , _50_ , 477–487. [CrossRef] [PubMed] 

9. Chtouki, K.; Rhanoui, M.; Mikram, M.; Yousfi, S.; Amazian, K. Supervised machine learning for breast cancer risk factors analysis and survival prediction. In _Proceedings of the International Conference on Big Data and Internet of Things_ ; Springer: Cham, Switzerland, 2022; pp. 59–71. 

10. Bera, K.; Schalper, K.A.; Rimm, D.L.; Velcheti, V.; Madabhushi, A. Artificial intelligence in digital pathology—new tools for diagnosis and precision oncology. _Nat. Rev. Clin. Oncol._ **2019** , _16_ , 703–715. [CrossRef] 

11. Shimizu, H.; Nakayama, K.I. Artificial intelligence in oncology. _Cancer Sci._ **2020** , _111_ , 1452. [CrossRef] 

12. Rhanoui, M.; Mikram, M.; Amazian, K.; Ait-Abderrahim, A.; Yousfi, S.; Toughrai, I. Multimodal Machine Learning for Predicting Post-Surgery Quality of Life in Colorectal Cancer Patients. _J. Imaging_ **2024** , _10_ , 297. [CrossRef] 

13. Dewaker, V.; Morya, V.K.; Kim, Y.H.; Park, S.T.; Kim, H.S.; Koh, Y.H. Revolutionizing oncology: The role of Artificial Intelligence (AI) as an antibody design, and optimization tools. _Biomark. Res._ **2025** , _13_ , 52. [CrossRef] [PubMed] 

14. Willemink, M.J.; Koszek, W.A.; Hardell, C.; Wu, J.; Fleischmann, D.; Harvey, H.; Folio, L.R.; Summers, R.M.; Rubin, D.L.; Lungren, M.P. Preparing medical imaging data for machine learning. _Radiology_ **2020** , _295_ , 4–15. [CrossRef] [PubMed] 

15. Belghiti, K.A.; Rekik, I.; Selim, S.; Mounia, M.; Rhanoui, M. Spatial Attention-Enhanced Diffusion Model for Multiple Sclerosis MRI Synthesis. In _Proceedings of the Meets Africa Workshop_ ; Springer: Cham, Switzerland, 2024; pp. 81–90. 

16. Das, S.; Nayak, S.P.; Sahoo, B.; Nayak, S.C. Machine learning in healthcare analytics: A state-of-the-art review. _Arch. Comput. Methods Eng._ **2024** , _31_ , 3923–3962. [CrossRef] 

17. Shen, D.; Wu, G.; Suk, H.I. Deep learning in medical image analysis. _Annu. Rev. Biomed. Eng._ **2017** , _19_ , 221–248. [CrossRef] 

18. Lai, M. Deep learning for medical image segmentation. _arXiv_ **2015** , arXiv:1505.02000. 

19. Caruana, R. Multitask learning. _Mach. Learn._ **1997** , _28_ , 41–75. [CrossRef] 

20. Zhu, X.; Suk, H.I.; Lee, S.W.; Shen, D. Subspace regularized sparse multitask learning for multiclass neurodegenerative disease identification. _IEEE Trans. Biomed. Eng._ **2015** , _63_ , 607–618. [CrossRef] 

21. Tabarestani, S.; Aghili, M.; Eslami, M.; Cabrerizo, M.; Barreto, A.; Rishe, N.; Curiel, R.E.; Loewenstein, D.; Duara, R.; Adjouadi, M. A distributed multitask multimodal approach for the prediction of Alzheimer’s disease in a longitudinal study. _NeuroImage_ **2020** , _206_ , 116317. [CrossRef] 

22. Yuan, H.; Paskov, I.; Paskov, H.; González, A.J.; Leslie, C.S. Multitask learning improves prediction of cancer drug sensitivity. _Sci. Rep._ **2016** , _6_ , 31619. [CrossRef] 

23. Collier, O.; Stoven, V.; Vert, J.P. LOTUS: A single-and multitask machine learning algorithm for the prediction of cancer driver genes. _PLoS Comput. Biol._ **2019** , _15_ , e1007381. [CrossRef] 

24. Zhou, C.; Ding, C.; Wang, X.; Lu, Z.; Tao, D. One-pass multi-task networks with cross-task guided attention for brain tumor segmentation. _IEEE Trans. Image Process._ **2020** , _29_ , 4516–4529. [CrossRef] 

25. Amyar, A.; Modzelewski, R.; Li, H.; Ruan, S. Multi-task deep learning based CT imaging analysis for COVID-19 pneumonia: Classification and segmentation. _Comput. Biol. Med._ **2020** , _126_ , 104037. [CrossRef] [PubMed] 

_Onco_ **2025** , _5_ , 34 

17 of 18 

26. Alom, M.Z.; Rahman, M.; Nasrin, M.S.; Taha, T.M.; Asari, V.K. COVID_MTNet: COVID-19 detection with multi-task deep learning approaches. _arXiv_ **2020** , arXiv:2004.03747. 

27. Li, J.; Zhao, G.; Tao, Y.; Zhai, P.; Chen, H.; He, H.; Cai, T. Multi-task contrastive learning for automatic CT and X-ray diagnosis of COVID-19. _Pattern Recognit._ **2021** , _114_ , 107848. [CrossRef] [PubMed] 

28. Mulyar, A.; Uzuner, O.; McInnes, B. MT-clinical BERT: Scaling clinical information extraction with multitask learning. _J. Am. Med. Inform. Assoc._ **2021** , _28_ , 2108–2115. [CrossRef] 

29. Zhang, Q.; Chen, S.; Liu, W. Balanced Knowledge Transfer in MTTL-ClinicalBERT: A Symmetrical Multi-Task Learning Framework for Clinical Text Classification. _Symmetry_ **2025** , _17_ , 823. [CrossRef] 

30. Zekaoui, N.E.; Rhanoui, M.; Yousfi, S.; Mikram, M. SSMT-PANBERT: A single-stage multitask model for phenotype extraction and assertion negation detection in unstructured clinical text. _Comput. Biol. Med._ **2025** , _195_ , 110651. [CrossRef] 

31. Ronneberger, O.; Fischer, P.; Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention, Munich, Germany, 5–9 October 2015; Springer: Berlin/Heidelberg, Germany, 2015; pp. 234–241. 

32. Simonyan, K.; Zisserman, A. Very deep convolutional networks for large-scale image recognition. _arXiv_ **2014** , arXiv:1409.1556. 

33. Deng, J.; Dong, W.; Socher, R.; Li, L.J.; Li, K.; Li, F.-F. Imagenet: A large-scale hierarchical image database. In Proceedings of the 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 20–25 June 2009; pp. 248–255. 

34. Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; Chen, L.C. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–23 June 2018; pp. 4510–4520. 

35. Zhang, Y.; Yang, Q. A survey on multi-task learning. _IEEE Trans. Knowl. Data Eng._ **2021** , _4_ , 5586–5609 [CrossRef] 

36. Ter-Sarkisov, A. One Shot Model For COVID-19 Classification and Lesions Segmentation In Chest CT Scans Using LSTM With Attention Mechanism. _medRxiv_ **2021** . 

37. Le, T.L.T.; Thome, N.; Bernard, S.; Bismuth, V.; Patoureaux, F. Multitask classification and segmentation for cancer diagnosis in mammography. _arXiv_ **2019** , arXiv:1909.05397. 

38. Yang, X.; Zeng, Z.; Yeo, S.Y.; Tan, C.; Tey, H.L.; Su, Y. A novel multi-task deep learning model for skin lesion segmentation and classification. _arXiv_ **2017** , arXiv:1703.01025. 

39. Gu, R.; Wang, G.; Song, T.; Huang, R.; Aertsen, M.; Deprest, J.; Ourselin, S.; Vercauteren, T.; Zhang, S. CA-Net: Comprehensive Attention Convolutional Neural Networks for Explainable Medical Image Segmentation. _IEEE Trans. Med Imaging_ **2021** , _40_ , 699–711. [CrossRef] [PubMed] 

40. Luo, C.; Shi, C.; Li, X.; Wang, X.; Chen, Y.; Gao, D.; Yin, Y.; Song, Q.; Wu, X.; Zhou, J. Multi-Task Learning Using Attention-Based Convolutional Encoder-Decoder for Dilated Cardiomyopathy CMR Segmentation and Classification. _Comput. Mater. Contin._ **2020** , _63_ , 995–1012. 

41. Foo, A.; Hsu, W.; Lee, M.L.; Lim, G.; Wong, T.Y. Multi-Task Learning for Diabetic Retinopathy Grading and Lesion Segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, New York, NY, USA, 7–12 February 2020; Volume 34, pp. 13267–13272. 

42. Gu, Z.; Cheng, J.; Fu, H.; Zhou, K.; Hao, H.; Zhao, Y.; Zhang, T.; Gao, S.; Liu, J. Ce-net: Context encoder network for 2d medical image segmentation. _IEEE Trans. Med. Imaging_ **2019** , _38_ , 2281–2292. [CrossRef] 

43. Lv, C.; Shu, X.J.; Liang, Q.; Qiu, J.; Xiong, Z.C.; bo Ye, J.; bo Li, S.; Liu, C.Q.; Niu, J.Z.; Chen, S.B.; et al. BrainTumNet: Multi-task deep learning framework for brain tumor segmentation and classification using adaptive masked transformers. _Front. Oncol._ **2025** , _15_ , 1585891. [CrossRef] 

44. Codella, N.; Rotemberg, V.; Tschandl, P.; Celebi, M.E.; Dusza, S.; Gutman, D.; Helba, B.; Kalloo, A.; Liopyris, K.; Marchetti, M.; et al. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). _arXiv_ **2019** , arXiv:1902.03368. 

45. Bulten, W.; Litjens, G.; Pinckaers, H.; Ström, P.; Eklund, M.; Kartasalo, K.; Demkin, M.; Dane, S. _The PANDA Challenge: Prostate cANcer graDe Assessment Using the Gleason Grading System_ ; Zenodo: Geneva, Switzerland, 2020. 

46. Selvaraju, R.R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; Batra, D. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 22–29 October 2017; pp. 618–626. 

47. Oktay, O.; Schlemper, J.; Folgoc, L.L.; Lee, M.; Heinrich, M.; Misawa, K.; Mori, K.; McDonagh, S.; Hammerla, N.Y.; Kainz, B.; et al. Attention u-net: Learning where to look for the pancreas. _arXiv_ **2018** , arXiv:1804.03999. 

48. Mehta, S.; Rastegari, M.; Caspi, A.; Shapiro, L.; Hajishirzi, H. Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation. In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018; pp. 552–568. 

_Onco_ **2025** , _5_ , 34 

18 of 18 

49. Bachmann, R.; Mizrahi, D.; Atanov, A.; Zamir, A. Multimae: Multi-modal multi-task masked autoencoders. In Proceedings of the European Conference on Computer Vision, Tel Aviv, Israel, 23–27 October 2022; pp. 348–367. 

50. Kufel, J.; Bielówka, M.; Rojek, M.; Mitr˛ega, A.; Lewandowski, P.; Cebula, M.; Krawczyk, D.; Bielówka, M.; Kondoł, D.; BargiełŁ ˛aczek, K.; et al. Multi-label classification of chest X-ray abnormalities using transfer learning techniques. _J. Pers. Med._ **2023** , _13_ , 1426. [CrossRef] 

**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 

