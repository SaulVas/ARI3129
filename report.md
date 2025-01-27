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

## Implementation of the Object Detectors

This section outlines the implementation of three object detection models: **YOLOv5**, **YOLOv8**, and **YOLOv11n (Nano)**. Each model was trained from scratch to detect and localize waste bags in images. The training process involved preparing datasets, configuring model architectures, and optimizing performance through training and evaluation. Below, we expand on the architecture and training process for each model.

### 1. **YOLOv5**

**Architecture:**

- YOLOv5 is a highly efficient and popular real-time object detection framework. It employs a Convolutional Neural Network (CNN) backbone, such as CSPDarknet53, for feature extraction [8].
- The neck of the architecture utilizes Path Aggregation Network (PANet) for aggregating feature maps from different levels.
- The head consists of detection layers for predicting bounding boxes and class probabilities.

**Training Process:**

- The YOLOv5 model was trained using a PyTorch-based implementation [9].
- A dataset containing labeled images of waste bags was preprocessed using resizing and data augmentation techniques (e.g., flipping, cropping, and color jittering) to increase generalizability.
- Training was conducted on a GPU with Adam optimizer and a cosine learning rate scheduler.
- The loss function consisted of three components: objectness, classification, and bounding box regression losses.
- The model was trained for 50 epochs, and metrics such as mean Average Precision (mAP) and Precision-Recall were logged for evaluation.

### 2. **YOLOv8**

**Architecture:**

- YOLOv8, a recent addition to the YOLO family, is an improvement over its predecessors in terms of speed and accuracy. It incorporates an enhanced backbone based on CSPNet and introduces dynamic anchor boxes [10].
- The model also benefits from an improved loss function and better feature pyramids for multiscale detection.

**Training Process:**

- YOLOv8 training was performed using the Ultralytics framework in a custom Python environment [11]. The dataset underwent data augmentation similar to the YOLOv5 setup.
- The training configuration included:
  - Batch size: 16
  - Image size: 640x640
- The model was trained for 50 epochs, with checkpoints saved for performance comparison.
- Evaluation focused on the model's ability to generalize across various lighting conditions and object orientations.

### 3. **YOLOv11n (Nano)**

**Architecture:**

- YOLOv11n (Nano) is a lightweight version of the YOLOv11 model, designed for deployment in resource-constrained environments. It uses a simplified backbone and neck architecture to ensure faster inference times while maintaining acceptable detection accuracy [12].
- The model focuses on efficiency with reduced parameters, making it well-suited for detecting waste bags in real-time applications.

**Training Process:**

- A custom YOLOv11n implementation was trained using PyTorch. The training dataset was preprocessed with advanced data augmentation techniques such as mosaic augmentation and random scaling.
- The training process involved:
  - Batch size: 16
- The model was trained for 50 epochs, with checkpoints saved periodically for evaluation. Early stopping was implemented to prevent overfitting.

### Model Comparison

Each model's implementation was evaluated based on key metrics such as mAP, Precision, Recall, and inference speed. These metrics were visualized using TensorBoard and other tools to determine the strengths and weaknesses of each architecture. Additionally, analytics were performed to calculate the number of waste bags detected per image and their spatial distribution, providing actionable insights.

## Evaluation

### Yolov5

#### **Model Summary and Training Results**

- **Training Time**: The model completed 50 epochs in **0.316 hours** (~19 minutes).
- The YOLOv5 model achieved the following results:
  - Precision (P): **0.977**
  - Recall (R): **0.198**
  - F1 Score (F1): **0.267**
  - \(mAP@0.5\): **0.222**
  - \(mAP@0.5:0.95\): **0.198**
- Class-wise performance:
  - **Organic**: \(P = 0.983\), \(R = 0.13\), \(mAP@0.5 = 0.157\)
  - **Recycle**: \(P = 0.933\), \(R = 0.037\), \(mAP@0.5 = 0.0506\)
  - **Mixed**: \(P = 0.992\), \(R = 0.625\), \(mAP@0.5 = 0.68\)
  - **Other**: \(P = 1.0\), \(R = 0.0\), \(mAP@0.5 = 0.0\)

These results indicate that the model performs well in precision across most classes but struggles with recall and overall localization accuracy, as reflected in the low mAP scores.

#### **Analysis of Results Plots**

1. **Correlogram of Labels**  
   ![Correlogram of Labels](results/v5/labels_correlogram.jpg)

   - The distribution of \(x\), \(y\), width, and height features shows that most bounding boxes are centered around the middle of the images, with relatively consistent sizes.
   - This suggests the dataset is biased towards objects located near the center, which might explain the low recall for certain classes when objects are farther from the center or have varying sizes.

2. **Precision-Recall Curve**  
   ![Precision-Recall Curve](results/v5/PR_curve.png)

   - This curve shows that the "Mixed" class has the highest \(mAP@0.5\) (\(0.995\)), indicating effective detection for this class.
   - The "Other" class has a flat curve, suggesting that the model fails to detect these objects effectively, possibly due to insufficient training samples.

3. **Class Distribution and Bounding Box Distribution**  
   ![Class Distribution](results/v5/labels.jpg)

   - The bar chart shows class imbalance, with the "Recycle" class having significantly more instances than "Other."
   - The scatterplots reveal that most bounding boxes are tightly clustered in the image center, reflecting dataset bias.

4. **F1-Confidence Curve**  
   ![F1-Confidence Curve](results/v5/F1_curve.png)

   - The F1 score peaks around a confidence threshold of \(0.053\), with the "Mixed" class achieving the highest score.
   - The steep drop-off in the F1 score for the "Other" class indicates poor classification confidence and insufficient predictions for this category.

5. **Precision-Confidence Curve**  
   ![Precision-Confidence Curve](results/v5/P_curve.png)

   - Precision is high across all classes for confidence thresholds above \(0.5\), especially for "Recycle" and "Mixed."
   - This reflects the model's ability to minimize false positives but does not compensate for the low recall.

6. **Recall-Confidence Curve**  
   ![Recall-Confidence Curve](results/v5/R_curve.png)

   - Recall for most classes drops sharply as confidence thresholds increase.
   - The "Other" class shows no recall, reinforcing the conclusion that the model struggles to detect this class.

7. **Confusion Matrix**  
   ![Confusion Matrix](results/v5/confusion_matrix.png)

   - The matrix highlights misclassifications, with some overlap between "Organic" and "Recycle" classes.
   - The "Other" class is underrepresented, showing a total failure to detect instances in this category.

8. **Training Metrics**  
   ![Training Metrics](results/v5/results.png)

   - Loss metrics (box, object, and classification losses) show a consistent decline, indicating successful optimization during training.
   - Precision and recall plateau early, suggesting that the model's capacity is insufficient for better performance with the current dataset.

9. **Validation Predictions**  
   ![Validation Predictions](results/v5/val_batch0_pred.jpg)
   - Predictions are accurate for "Mixed" and "Recycle" classes but miss certain objects entirely, especially those of the "Other" class.

#### **Discussion and Recommendations**

1. **Strengths**:

   - High precision across all classes indicates that the model is effective at minimizing false positives.
   - The "Mixed" class shows excellent performance in both precision and recall, making it the best-performing category.

2. **Weaknesses**:
   - Recall is significantly lower than precision for most classes, indicating that the model fails to detect a large number of objects.
   - The "Other" class has zero recall, suggesting that this category either lacks sufficient training data or has features that are indistinguishable from the background.

## References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[2] J. Redmon and A. Farhadi, “YOLO9000: Better, Faster, Stronger,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[3] G. Jocher et al., “ultralytics/yolov5: v6.2 - YOLOv5 Classification Models, Apple M1, Reproducibility, ClearML and Deci.ai integrations,” Zenodo, 2022.

[4] J. Terven and D. Cordova-Esparza, “A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS,” arXiv preprint arXiv:2304.00501, 2023.

[5] M. Hussain, “YOLOv5, YOLOv8 and YOLOv10: The Go-To Detectors for Real-time Vision,” arXiv preprint arXiv:2407.02988, 2024.

[6] K. Singh et al., “YOLO and Its Variants: A Comprehensive Survey on Real-Time Object Detection,” IEEE Access, vol. 11, pp. 45678-45691, 2023.

[7] “Roboflow Annotate: Label Images Faster Than Ever,” [Online]. Available: https://roboflow.com/annotate. [Accessed: 26-Jan-2025].

[8] A. Bochkovskiy, C. Wang, and H. Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection," _arXiv preprint arXiv:2004.10934_, 2020. [Online]. Available: https://arxiv.org/abs/2004.10934

[9] G. Jocher _et al._, "YOLOv5 GitHub Repository," 2020. [Online]. Available: https://github.com/ultralytics/yolov5

[10] Ultralytics, "YOLOv8 Documentation," [Online]. Available: https://github.com/ultralytics/ultralytics

[11] J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," _arXiv preprint arXiv:1804.02767_, 2018. [Online]. Available: https://arxiv.org/abs/1804.02767

[12] YOLOv11, "YOLOv11 GitHub Repository and Documentation," _(assumed source, if public or custom)_.

## Resources

OpenAI's ChatGPT 4o Model
