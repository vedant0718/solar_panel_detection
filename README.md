# Solar Panel Detection

This repository contains a complete pipeline for detecting solar panels using a YOLO-based object detection model. The project is designed to explore a dataset, implement core evaluation metrics, train a model, evaluate the model's performance, and interactively visualize results through a modern Streamlit dashboard. The dashboard is styled with a dark theme using black and orange (Streamlit's default accent colors) for a sleek, modern look.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
  - [Running Individual Scripts](#running-individual-scripts)

## Project Overview

The **Solar Panel Detection** project aims to:
- **Explore the Dataset:**  
  Compute dataset statistics such as total instances of solar panels, label counts per image, and distribution of solar panel areas. Generate histograms with observations.
- **Implement Fundamental Functions:**  
  - Compute Intersection over Union (IoU) using the Shapely library.
  - Compute Average Precision (AP) using three methods:
    - Pascal VOC 11-point interpolation.
    - COCO 101-point interpolation.
    - Area under the Precision-Recall curve.
  - Generate synthetic test data.
  - Calculate precision and recall metrics.
- **Train the Model:**  
  Use a YOLO-based model (e.g., `yolov8n.pt`) to train on the provided dataset configuration (`data.yaml`), and save the best model weights.
- **Evaluate the Model:**  
  Visualize predictions on test images, overlaying predicted bounding boxes (in red) and ground truth boxes (in green). Compute evaluation metrics (precision, recall, F1-score) using the supervision library and present the results in tables.
- **Split the Dataset:**  
  Automatically split the dataset into training, validation, and test sets.
- **Interactive Dashboard:**  
  Run an interactive Streamlit dashboard that allows users to easily explore each module with minimal scrolling and a modern, intuitive interface.

## Features

- **Data Exploration:**
  - Calculate the total number of solar panel instances.
  - Plot a histogram of solar panel areas (in m²) with accompanying observations.
  - Display a table showing the number of labels per image.
- **Fundamental Functions:**
  - Compute IoU between bounding boxes using the Shapely library.
  - Implement Average Precision (AP) using VOC 11-point, COCO 101-point, and area under PR curve methods.
  - Generate synthetic test data for evaluation.
  - Compute precision-recall curves.
- **Model Training:**
  - Train a YOLO object detection model on the dataset.
  - Save the best model weights to the specified folder.
- **Model Evaluation:**
  - Visualize predictions on test images with ground truth and predicted bounding boxes.
  - Evaluate model performance using confusion matrices and compute precision, recall, and F1-scores.
- **Dataset Splitting:**
  - Automatically split images and corresponding label files into training, validation, and test sets.
- **Interactive Streamlit Dashboard:**
  - Use a modern dark-themed UI with black and orange accents.
  - Navigate between modules via a sidebar and view results in tabs and expandable cards.
  - Minimal scrolling with a clean, intuitive layout.

## Project Structure

```
solar_panel_detection/
├── dataset/
│   ├── images/
│   │   ├── native/   # Original high-resolution images (31 cm native resolution resized to 416x416)
│   │   ├── train/    # Training images (after splitting)
│   │   ├── val/      # Validation images (after splitting)
│   │   └── test/     # Test images (after splitting)
│   └── labels/
│       ├── labels_native/  # Original labels in MS-COCO format (horizontal bounding boxes)
│       ├── train/          # Training labels
│       ├── val/            # Validation labels
│       └── test/           # Test labels
├── results/
│   ├── plots/        # Generated plots (e.g., histograms)
│   ├── metrics/      # Metrics reports (e.g., AP scores, data stats, metrics tables)
│   └── models/
│       └── solar_panel_yolo/
│           └── weights/    # Trained model weights (e.g., best.pt)
├── data_exploration.py     # Script for computing dataset statistics and visualizations
├── fundamental_functions.py # Core functions: IoU, AP calculations, synthetic test data generation, precision-recall
├── model_training.py       # Script to train the YOLO model
├── model_evaluation.py     # Script to evaluate the model and visualize predictions
├── split_dataset.py        # Script to split the dataset into train, validation, and test sets
├── streamlit_app.py        # Interactive Streamlit dashboard connecting all modules
├── utils.py                # Utility functions (e.g., drawing bounding boxes, format conversion)
├── requirements.txt        # List of required Python packages
└── README.md               # This file
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/vedant0718/solar_panel_detection.git
   cd solar_panel_detection
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit Dashboard

To start the interactive dashboard, run:

```bash
streamlit run streamlit_app.py
```

The dashboard provides access to:
- **Data Exploration:** View the histogram of solar panel areas (with observations) and a table of label counts per image.
- **Fundamental Functions:** Test IoU and AP calculations, and view precision-recall curves.
- **Model Training:** Train the YOLO model.
- **Model Evaluation:** Visualize predictions on test images and evaluate model performance with detailed metrics.
- **Dataset Splitting:** Split your dataset into training, validation, and test sets.

### Running Individual Scripts

Alternatively, you can run specific modules directly:

- **Data Exploration:** `python data_exploration.py`
- **Model Training:** `python model_training.py`
- **Model Evaluation:** `python model_evaluation.py`
- **Dataset Splitting:** `python split_dataset.py`