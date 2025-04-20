# Emotion Detection 

This project focuses on classifying emotions expressed on a person’s face into one of seven predefined categories using deep convolutional neural networks (CNN). The model is built to process images from the FER-2013 dataset, which contains a large set of labeled facial expressions. These images are 48x48 pixels in size and are grayscale, allowing the model to focus on the key features of the face.

The FER-2013 dataset, collected from various sources, contains over 35,000 images, and each image is tagged with one of the following emotions:

- **Angry**: A facial expression characterized by furrowed brows, a tight mouth, and intense, glaring eyes.
- **Disgusted**: A face that shows a wrinkled nose, pursed lips, and raised upper lip, signaling displeasure or revulsion.
- **Fearful**: Eyes wide open and eyebrows raised, with a tense and nervous expression, reflecting fear or anxiety.
- **Happy**: A smiling expression with eyes slightly squinted, signifying joy or happiness.
- **Neutral**: A neutral or blank expression with little to no visible emotions, showing neither positive nor negative feelings.
- **Sad**: Drooping eyes, a frown, and slumped features, indicating a feeling of sadness or sorrow.
- **Surprised**: Wide-open eyes and raised eyebrows, usually accompanied by a slightly open mouth, showing shock or amazement.

The deep CNN model learns to identify and classify these emotions by training on the large set of labeled images, allowing it to detect subtle variations in facial expressions with high accuracy. By processing these images, the model is capable of predicting the correct emotion based on the person’s facial features, making it suitable for real-time emotion detection applications.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Basic Usage](#basic-usage)
- [Data Preparation (optional)](#data-preparation-optional)
- [Algorithm](#algorithm)
- [Folder Structure](#folder-structure)
- [Logging and Plotting](#logging-and-plotting)
- [References](#references)
- [Made by Shubham Malik](#made-by-shubham-malik)

## Introduction

This project aims to detect emotions from faces in real-time using a deep CNN model. The model uses the FER-2013 dataset, published at the International Conference on Machine Learning (ICML), which contains 35,887 images.

## Dependencies

This project requires the following Python libraries:

- Python 3
- OpenCV
- TensorFlow 2.0
- Keras (included in TensorFlow)
- Matplotlib (for graph plotting)

To install the required packages, run the following command:
```
pip install -r requirements.txt

```

## Basic Usage

Follow the instructions below to use this repository:

1. **Clone the repository** and enter the folder:

    ```bash
    git clone https://github.com/subhm2004/Face_Detection_Using_Deep_Learning.git
    cd Face_Detection_Using_Deep_Learning
    ```

2. **Download the FER-2013 dataset** and place it inside the `src` folder.

3. To **train the model** from scratch, run:

    ```bash
    cd src
    python emotions.py --mode train
    ```

4. To **view the predictions** without retraining, download the pre-trained model (available in the repository) and run:

    ```bash
    cd src
    python emotions.py --mode display
    ```



## Data Preparation (optional)

The original FER-2013 dataset is available in CSV format. To use this dataset, you may need to convert the CSV file to individual image files (PNG format) or perform other preprocessing. The following scripts are provided for your convenience:

### 1. **`convert_er2013_to_image.py`**

This script can convert the FER-2013 dataset from CSV format into individual image files (PNG format). To convert the dataset, run the following command:

```bash
cd src
python convert_er2013_to_image.py
```
## Data Preparation (optional)

### 1. **`convert_er2013_to_image.py`**

This script processes the CSV file and converts it into 48x48 PNG images that can be used for training or testing.

### 2. **`dataset_prepare.py`**

This script is used to preprocess the FER-2013 dataset if you're working with data in CSV format. It includes various data preparation steps like resizing, normalization, and splitting into training/testing sets. To use it, run:

```bash
cd src
python dataset_prepare.py
```
You can customize the preprocessing steps in this script based on your dataset and requirements.


## Algorithm

### 1. **Face Detection**: 
The algorithm uses Haar cascades to detect faces in each frame of the webcam feed.

### 2. **Preprocessing**: 
The region containing the face is resized to 48x48 pixels and passed to the CNN as input.

### 3. **CNN Model**: 
The CNN model outputs a list of softmax scores for the seven emotion classes.

### 4. **Prediction**: 
The emotion with the highest score is displayed on the screen.

---
## Folder Structure
```
Emotion-Detection-Deep-Learning/
│
├── src/                        # Main source code
│   ├── emotions.py             # Main script for training and prediction
│   ├── convert_er2013_to_image.py  # Converts the FER-2013 dataset from CSV to image files
│   ├── dataset_prepare.py      # Preprocessing script for FER-2013 dataset
│   ├── plot_graph_and_pie.py   # Script for plotting accuracy and loss graphs
│   ├── model/                  # Contains model architecture and utility files
│   │   ├── cnn_model.py        # CNN model definition
│   │   └── utils.py            # Helper functions (e.g., for image processing)
│   └── logs/                   # Logs for training progress
│       └── training_log.txt    # Log of training progress
│
├── data/                       # Data-related files
│   ├── fer2013.csv             # FER-2013 dataset in CSV format
│   └── images/                 # Folder to store processed images
│       └── train/              # Training images
│       └── test/               # Test images
│
├── pretrained_models/          # Pre-trained models
│   └── emotion_detection_model.h5  # Pre-trained emotion detection model file
│
├── requirements.txt            # List of required Python libraries
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file to avoid versioning unnecessary files
```


## Logging and Plotting

### **Training Log**
During training, the model logs the training process (epoch, accuracy, loss, validation accuracy, and validation loss) in a file named `training_log.txt`. This log is updated after each epoch and is stored in the `src` folder.

The training log is created using the following code in the `emotions.py` file:

```python
csv_logger = CSVLogger('training_log.txt', append=True)
```
Plotting Graphs
After training, you can visualize the accuracy and loss curves by running the plot_graph_and_pie.py script. This will display graphs showing the training and validation accuracy, as well as the training and validation loss.

To plot the graphs, run:
```
python plot_graphs_and_pie.py
```
This script will read the data from the `training_log.txt` file and display the following graphs:

- **Training and validation accuracy over epochs**
- **Training and validation loss over epochs**
- **A pie chart of the emotion class distribution** (if desired)

---


## References

- **FER-2013 Dataset**: The FER-2013 dataset is used to train the model and is available at [Kaggle FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
  
- **Paper Reference**:  
  "Challenges in Representation Learning: A report on three machine learning contests."  
  I. Goodfellow, D. Erhan, P.L. Carrier, A. Courville, M. Mirza, B. Hamner, W. Cukierski, Y. Tang, D.H. Lee, Y. Zhou, C. Ramaiah, F. Feng, R. Li, X. Wang, D. Athanasakis, J. Shawe-Taylor, M. Milakov, J. Park, R. Ionescu, M. Popescu, C. Grozea, J. Bergstra, J. Xie, L. Romaszko, B. Xu, Z. Chuang, Y. Bengio. arXiv 2013.


---
Made with ❤️ by Shubham Malik





