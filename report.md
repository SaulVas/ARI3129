# ARI3129 Advanced Computer Vision for AI

Saul Vassallo, Nick Gaerty, Luca D'Ascari

## Introduction

## Background

Object detection is a key task in computer vision, involving the simultaneous classification and localisation of objects within an image. It underpins numerous real-world applications, from autonomous vehicles to healthcare and environmental monitoring. The goal is to detect objects with high precision and speed, making it a vital component of intelligent systems. The advent of deep learning models such as YOLO (You Only Look Once) has significantly advanced object detection by balancing accuracy with real-time performance [1].

The YOLO series of models has revolutionised object detection since its introduction in 2016. Unlike traditional methods, which involve separate processes for region proposal and classification, YOLO frames detection as a single regression problem, predicting bounding boxes and class probabilities in one pass through the network. This unified approach ensures speed and efficiency, enabling YOLO to process images at impressive frame rates while maintaining competitive accuracy [2]. Over the years, enhancements in the YOLO architecture have introduced features such as anchor boxes, multi-scale predictions, and better loss functions, which have improved its robustness and scalability [3].

Recent iterations like YOLOv5, YOLOv8, and YOLOv11 have further refined the model’s capabilities. YOLOv5 incorporates automated anchor generation and mosaic data augmentation, making it highly adaptable to diverse datasets [4]. YOLOv8, on the other hand, integrates advanced feature aggregation and attention mechanisms, boosting its performance on complex datasets [5]. The YOLOv11 version builds on these advancements by adopting state-of-the-art training strategies and expanding its ability to handle challenging scenarios, such as overlapping objects and occlusions [6]. These models are widely regarded for their ability to balance computational efficiency with high detection accuracy, making them ideal for time-sensitive applications.

In addition to detection algorithms, effective dataset preparation is critical to achieving reliable results. Roboflow, a widely used platform in the computer vision community, simplifies the end-to-end dataset creation pipeline. Its powerful annotation tools leverage AI to speed up labeling while maintaining accuracy, a crucial step in training high-performing models. Furthermore, Roboflow offers built-in data augmentation techniques, such as flipping, rotation, brightness adjustments, and cropping, to increase dataset diversity and improve model generalization. These augmentations simulate real-world conditions, ensuring that models trained on augmented datasets are robust across various scenarios [7].

Roboflow also supports seamless exporting of datasets in formats compatible with machine learning frameworks such as TensorFlow, PyTorch, and YOLO, facilitating the training process. Additionally, its integration with open-source datasets provides access to a vast repository of pre-labeled data, enabling researchers to augment their datasets further without starting from scratch. These features collectively make Roboflow a cornerstone for modern object detection workflows.

Together, the YOLO models and Roboflow form a synergistic pipeline for object detection. While the YOLO family offers cutting-edge detection capabilities, Roboflow ensures that the datasets feeding these models are optimised for success. This combination addresses challenges such as class imbalance, data diversity, and model scalability, enabling high accuracy and robust performance in real-world applications.

## Implementation

## Evaluation

## References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[2] J. Redmon and A. Farhadi, “YOLO9000: Better, Faster, Stronger,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[3] G. Jocher et al., “ultralytics/yolov5: v6.2 - YOLOv5 Classification Models, Apple M1, Reproducibility, ClearML and Deci.ai integrations,” Zenodo, 2022.

[4] J. Terven and D. Cordova-Esparza, “A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS,” arXiv preprint arXiv:2304.00501, 2023.

[5] M. Hussain, “YOLOv5, YOLOv8 and YOLOv10: The Go-To Detectors for Real-time Vision,” arXiv preprint arXiv:2407.02988, 2024.

[6] K. Singh et al., “YOLO and Its Variants: A Comprehensive Survey on Real-Time Object Detection,” IEEE Access, vol. 11, pp. 45678-45691, 2023.

[7] “Roboflow Annotate: Label Images Faster Than Ever,” [Online]. Available: https://roboflow.com/annotate. [Accessed: 26-Jan-2025].

## Resources

OpenAI's ChatGPT 4o Model
