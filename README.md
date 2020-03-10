# RSNA_Intracranial-hemorrhage

### Project description:

Hemorrhage in the brain (Intracranial Hemorrhage) is one of the top five fatal health problems. The hemorrhage causes bleeding inside the skull (typically known as cranium). It accounts for approximately 10% of strokes in the U.S thus diagnosing it quickly and efficiently is of utmost importance.

Intracranial Hemorrhage can have many consequences depending on the location, size, type of bleed in the brain. The relationship is not that simple to interpret. A small hemorrhage might be equally fatal as a big one if it is in a critical location. If a patient shows acute hemorrhage symptoms such as loss of consciousness then an immediate diagnosis is required to carry out the initial tests but the procedure is very complicated and often time-consuming. The delay in diagnosis and treatment can have severe effects on patients, especially the ones in need of immediate attention. The solution proposed in this project aims to make this process efficient by automating the preliminary step to identify the location, type, and size of the bleed, the doctors can provide a better diagnosis and a faster one as the work of identification is done by the algorithm. Also, the algorithm treats every scan with the same scrutiny, reducing human error and learning more as it goes.

Thus, the challenge is to build an algorithm to detect acute intracranial hemorrhage and its subtypes. This project will be a step towards identifying each of these relevant attributes for a better diagnosis.

Check the image below for more information on the various subtypes.

![test]()

### Approach:

We use [Jeremy Howard's clean dataset](https://www.kaggle.com/jhoward/rsna-hemorrhage-jpg) instead of the [original dataset](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) because the original does not fit on Google Colab's disk space. An in depth analysis of how he created this dataset is [shown here](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai). Jeremy's dataset has an image size of (256,256) and the metadata stored in data frames. For any troubles downloading the dataset, refer to [this link](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109520)

### Papers read:

[1] 

### Reference kernels:
