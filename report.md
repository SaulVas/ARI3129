# ARI3129 Advanced Computer Vision for AI

Saul Vassallo, Nick Gaerty, Luca D'Ascari

## Introduction

Waste management is a growing challenge in urban areas, with improper disposal leading to environmental pollution and public health concerns. Efficient waste collection and categorization are crucial for maintaining clean cities and promoting recycling efforts. In recent years, computer vision has emerged as a valuable tool in automating waste detection and classification, offering the potential to streamline waste management operations and improve efficiency.

This project focuses on the analysis of domestic waste collection on Maltese streets using computer vision techniques. The goal is to develop an object detection system capable of identifying and classifying different types of waste bags, such as mixed waste, organic waste, and recyclable materials. The project is structured into several key phases, starting with the collection and annotation of images, followed by data augmentation to enhance the dataset, and culminating in the training of object detection models using the YOLO framework, specifically versions YOLOv5,
YOLOv8, and YOLOv11. The final models will be evaluated to assess their effectiveness in accurately detecting and categorizing waste bags.

Through this project, we aim to demonstrate how artificial intelligence can be leveraged to support sustainable waste management practices, offering automated solutions that can contribute to cleaner and more efficient waste collection systems.

## Background

Object detection is a key task in computer vision, involving the simultaneous classification and localisation of objects within an image. It underpins numerous real-world applications, from autonomous vehicles to healthcare and environmental monitoring. The goal is to detect objects with high precision and speed, making it a vital component of intelligent systems. The advent of deep learning models such as YOLO (You Only Look Once) has significantly advanced object detection by balancing accuracy with real-time performance [1].

The YOLO series of models has revolutionised object detection since its introduction in 2016. Unlike traditional methods, which involve separate processes for region proposal and classification, YOLO frames detection as a single regression problem, predicting bounding boxes and class probabilities in one pass through the network. This unified approach ensures speed and efficiency, enabling YOLO to process images at impressive frame rates while maintaining competitive accuracy [2]. Over the years, enhancements in the YOLO architecture have introduced features such as anchor boxes, multi-scale predictions, and better loss functions, which have improved its robustness and scalability [3].

Recent iterations like YOLOv5, YOLOv8, and YOLOv11 have further refined the model’s capabilities. YOLOv5 incorporates automated anchor generation and mosaic data augmentation, making it highly adaptable to diverse datasets [4]. YOLOv8, on the other hand, integrates advanced feature aggregation and attention mechanisms, boosting its performance on complex datasets [5]. The YOLOv11 version builds on these advancements by adopting state-of-the-art training strategies and expanding its ability to handle challenging scenarios, such as overlapping objects and occlusions [6]. These models are widely regarded for their ability to balance computational efficiency with high detection accuracy, making them ideal for time-sensitive applications.

In addition to detection algorithms, effective dataset preparation is critical to achieving reliable results. Roboflow, a widely used platform in the computer vision community, simplifies the end-to-end dataset creation pipeline. Its powerful annotation tools leverage AI to speed up labeling while maintaining accuracy, a crucial step in training high-performing models. Furthermore, Roboflow offers built-in data augmentation techniques, such as flipping, rotation, brightness adjustments, and cropping, to increase dataset diversity and improve model generalization. These augmentations simulate real-world conditions, ensuring that models trained on augmented datasets are robust across various scenarios [7].

Roboflow also supports seamless exporting of datasets in formats compatible with machine learning frameworks such as TensorFlow, PyTorch, and YOLO, facilitating the training process. Additionally, its integration with open-source datasets provides access to a vast repository of pre-labeled data, enabling researchers to augment their datasets further without starting from scratch. These features collectively make Roboflow a cornerstone for modern object detection workflows.

Together, the YOLO models and Roboflow form a synergistic pipeline for object detection. While the YOLO family offers cutting-edge detection capabilities, Roboflow ensures that the datasets feeding these models are optimised for success. This combination addresses challenges such as class imbalance, data diversity, and model scalability, enabling high accuracy and robust performance in real-world applications.

## Data Preparation

The data preparation phase was a crucial aspect of our project, as it formed the foundation for training an accurate and unbiased object detection model. Task 1 of the assignment required us to collect, annotate, and prepare a dataset of domestic waste bags categorized into three types: mixed (black bags), recyclable (grey bags), and organic (white bags). Our goal was to ensure that the dataset was well-structured, balanced, and representative of real-world conditions, which would enable the model to generalize effectively across different environments. This section outlines the steps taken to collect, clean, annotate, and prepare the dataset for training.

**Image Collection**

To achieve a balanced dataset, we decided to collect 75 images per category, resulting in an initial dataset of 225 images. The decision to have an equal number of images for each waste category was made to prevent bias during model training, ensuring fair representation and performance across all types of waste.

The images were captured from various locations, including San Ġwann, Sliema, Swieqi, and Gżira, to introduce diversity in environmental conditions such as lighting, background, and terrain. Capturing images at different times of the day was also a key consideration, as waste collection
schedules vary across locations. To maximize the number of available waste bags, we adhered to the local waste collection schedule:

* **Organic waste (white bags):** Monday, Wednesday, and Friday.
* **Mixed waste (black bags):** Tuesday and Saturday.
* **Recycling (grey bags):** Thursday.

In areas such as Sliema, where collection occurs later in the day, images were taken during sunset, whereas for other locations, images were captured in the morning or early afternoon to align with pickup schedules. This approach ensured varied lighting conditions, which is beneficial for the model's ability to recognize waste bags in different real-world scenarios.

Additionally, to enhance efficiency during data collection, multiple images were taken of the same waste bag from different angles and backgrounds, simulating diverse environments. Another innovative approach involved carrying a waste bag to different locations and capturing images after
repositioning or slightly altering its shape to mimic natural variation.

**Image Cleaning and Organization**

Once the images were collected, they were organized into their respective categories: Organic, Mixed, and Recycling. Before proceeding with annotation, an image cleaning process was conducted. This involved reviewing all images to identify and obscure any personally identifiable
information, such as car number plates or human faces. Anything detected was covered with white markings using image editing tools to anonymize the data. Fortunately, no images contained human faces, simplifying the cleaning process.

Next, to ensure consistency in data processing, all images were resized to a standard resolution of 4032x3024 pixels. Initially, this resizing was done manually using macOS's built-in resizing tool, as we were unaware that Roboflow provided an automatic resizing option. We also ensured every image was in the same format, opting for the .jpg files. These steps ensured uniformity across the dataset, which is essential for efficient model training and evaluation.

**Annotation Process**

Following the data cleaning phase, the images were uploaded to Roboflow, a popular platform for dataset annotation and augmentation. Each image was meticulously reviewed, and bounding boxes were manually drawn around waste bags. The waste bags were classified into four classes:

* Organic (white bags)
* Mixed (black bags)
* Recycling (grey bags)
* Other (any waste that does not fit into the above categories)

To ensure high-quality annotations, careful attention was given to the accuracy of the bounding boxes, as incorrect labelling could negatively impact the model's performance.

**Data Augmentation**

Once annotation was complete, data augmentation was applied using Roboflow's built-in tools. Augmentation is essential for increasing dataset diversity and improving model robustness. The following augmentation techniques were applied:

* **Flipping:** Both horizontal and vertical flips to simulate different orientations.
* **Cropping:** Random cropping to introduce variation in positioning.
* **Zooming:** Adjusting image zoom levels.
* **Saturation:** Varying saturation levels between -25% and 25%.
* **Exposure:** Modifying exposure between -10% and 10% to simulate different lighting
  conditions.
* **Blur:** Adding Gaussian blur up to 1.5px to replicate environmental conditions.

After augmentation, the dataset size increased to 543 images,
which were split into training, validation, and test sets in a 70/20/10 ratio:

* **Train set:** 447 images
* **Validation set:** 45 images
* **Test set:** 21 images

This split was chosen to ensure the model had sufficient data for training while retaining enough samples for performance evaluation.

**Dataset Export and Preparation for Model Training**

Once the annotation and augmentation process was completed, the dataset was exported in three formats, each corresponding to a different version of the YOLO (You Only Look Once) object detection framework:

* **YOLOv5**
* **YOLOv8**
* **YOLOv11**

Exporting the dataset in multiple YOLO versions provided flexibility for experimentation and comparison in the subsequent training phase. These versions were selected to explore the improvements in detection accuracy and performance across different iterations of the YOLO framework.

This structured approach to data preparation ensured that our dataset was well-balanced, diverse, and ready for object detection model training. The comprehensive cleaning, annotation, and augmentation steps helped create a robust dataset capable of supporting accurate waste bag detection in varied environments.

## Implementation

## Evaluation

## References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[2] J. Redmon and A. Farhadi, “YOLO9000: Better, Faster, Stronger,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[3] G. Jocher et al., “ultralytics/yolov5: v6.2 - YOLOv5 Classification Models, Apple M1, Reproducibility, ClearML and Deci.ai integrations,” Zenodo, 2022.

[4] J. Terven and D. Cordova-Esparza, “A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS,” arXiv preprint arXiv:2304.00501, 2023.

[5] M. Hussain, “YOLOv5, YOLOv8 and YOLOv10: The Go-To Detectors for Real-time Vision,” arXiv preprint arXiv:2407.02988, 2024.

[6] K. Singh et al., “YOLO and Its Variants: A Comprehensive Survey on Real-Time Object Detection,” IEEE Access, vol. 11, pp. 45678-45691, 2023.

[7] “Roboflow Annotate: Label Images Faster Than Ever,” [Online]. Available: https://roboflow.com/annotate. [Accessed: 26-Jan-2025].

## Resources

OpenAI's ChatGPT 4o Model
