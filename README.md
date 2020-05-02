# RSNA_Intracranial-hemorrhage

### Table of Contents:
- [Introduction](#intro)
- [Data description](#dd)
- [Notebooks](#nbs)
  * [00_setup.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/00_setup.ipynb)
  * [01_data_cleaning.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/01_data_cleaning.ipynb)
  * [02_data_exploration.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/02_data_exlporation.ipynb)
  * [03_data_augmentation.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/03_data_augmentation.ipynb)
  * [04_baseline_model](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/04_baseline_model.ipynb)
  * [05-metadata.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/05-metadata.ipynb)
  * [06_multilabel_classification.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/tree/master/nbs/06_multilabel_classification.ipynb)
  * [07a_diff_archs_resnet101.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/07a_diff_archs_resnet101.ipynb)
  * [07b_diff_archs_densenet121.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/07b_different_archs_densenet121.ipynb)
  * [07c_diff_archs_alexnet.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/07c_diff_archs_alexnet.ipynb)
  * [09_mixed_precision.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/09_mixed_precision.ipynb)
  * [10_progressive_resizing.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/10_progressive_resizing.ipynb)
- [Papers Read](#pr)
- [References](#ref)

<a name='intro'></a>
### Introduction:

Hemorrhage in the brain (Intracranial Hemorrhage) is one of the top five fatal health problems. The hemorrhage causes bleeding inside the skull (typically known as cranium). It accounts for approximately 10% of strokes in the U.S thus diagnosing it quickly and efficiently is of utmost importance.

Intracranial Hemorrhage can have many consequences depending on the location, size, type of bleed in the brain. The relationship is not that simple to interpret. A small hemorrhage might be equally fatal as a big one if it is in a critical location. If a patient shows acute hemorrhage symptoms such as loss of consciousness then an immediate diagnosis is required to carry out the initial tests but the procedure is very complicated and often time-consuming. The delay in diagnosis and treatment can have severe effects on patients, especially the ones in need of immediate attention. The solution proposed in this project aims to make this process efficient by automating the preliminary step to identify the location, type, and size of the bleed, the doctors can provide a better diagnosis and a faster one as the work of identification is done by the algorithm. Also, the algorithm treats every scan with the same scrutiny, reducing human error and learning more as it goes.

Thus, the challenge is to build an algorithm to detect acute intracranial hemorrhage and its subtypes. This project will be a step towards identifying each of these relevant attributes for a better diagnosis.

Check the image below for more information on the various subtypes.

![test](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/images/subtypes.png)

<a name='dd'></a>
### Data description

We use [Jeremy Howard's clean dataset](https://www.kaggle.com/jhoward/rsna-hemorrhage-jpg) instead of the [original dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) An in depth analysis of how he created this dataset is [shown here](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai). Jeremy's dataset has an image size of (256,256) and the metadata stored in data frames. For any troubles downloading the original dataset, refer to [this link](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109520)

The data is provided in 3 parts, the dicom images that look as follows

![test](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/images/data_sample.png)

metadata such as Patient ID, Image position, Image Orientation, window center, window width and so on. And a target data frame mapping every patient to the hemorrhage.

![test](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/images/sample_csv.png)


For domain knowledge, refer to [readiology masterclass](https://www.radiologymasterclass.co.uk/tutorials/ct/ct_acute_brain/ct_brain_cerebral_haemorrhage)

<a name='nbs'></a>
### Notebooks:

The kernels are arranged in numerical order so that people can see where we started and how we went about our experiments

[00_setup.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/00_setup.ipynb) - how to setup and download the dataset on Google colab.

[01_data_cleaning.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/01_data_cleaning.ipynb) - steps followed to clean the data

[02_data_exploration.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/02_data_exlporation.ipynb) - exploratory data analysis on the images and csv files

[03_data_augmentation.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/03_data_augmentation.ipynb) - applying various transforms on the images and seeing which ones are useful.

[04_baseline_model](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/04_baseline_model.ipynb) - train a resnet18 with and without pretraining.

[05-metadata.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/05-metadata.ipynb) - checking the usability of the metadata in training the model. 

[06_multilabel_classification.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/tree/master/nbs/06_multilabel_classification.ipynb) - We take a subset of the dataframe that includes the images with bleeds present. We create a new column in the dataframe that includes all the sub categories of bleeds present in an image separated by semicolon. We then train a model on this data and use a threshold of 0.5 to make predictions. The threshold is set to a high value because we want to be conservative with medical data and not predict a bleed unless we are very sure.

[07a_diff_archs_resnet101.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/07a_diff_archs_resnet101.ipynb) - We now try different architectures to improve our model's performance. One way to do so is to use more layers. Hence we try resnet101 after resnet50. It does give a higher accuracy however not so significant.

[07b_diff_archs_densenet121.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/07b_different_archs_densenet121.ipynb) - Anther good architecture for image classification is densenet. Google it to find more information about this architecture.

[07c_diff_archs_alexnet.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/07c_diff_archs_alexnet.ipynb) - Google to learn more about the alexnet architecture.

[09_mixed_precision.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/09_mixed_precision.ipynb)


[10_progressive_resizing.ipynb](https://github.com/dipam7/RSNA_Intracranial-hemorrhage/blob/master/nbs/10_progressive_resizing.ipynb)

<a name='pr'></a>
### Readings:

[1] [Development and Validation of Deep Learning Algorithms for Detection of Critical Findings in Head CT Scans](https://arxiv.org/pdf/1803.05854.pdf)

[2] [Qure.ai blog](http://blog.qure.ai)

[3] [Expert-level detection of acute intracranial hemorrhage on head computed tomography using deep learning](https://www.pnas.org/content/116/45/22737)

[4] [Extracting 2D weak labels from volume labels using multiple instance learning in CT hemorrhage detection](https://arxiv.org/pdf/1911.05650.pdf)

[5] [fastai—A Layered API for Deep Learning](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/)

[6] [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)

[7] [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)

[8] [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)

<a name='ref'></a>
### References:

[1. Jeremy's notebooks](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/114214)

[2. Basic EDA + Data Visualization](https://www.kaggle.com/marcovasquez/basic-eda-data-visualization)

[3. See like a radiologist with systematic windowing](https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing)
