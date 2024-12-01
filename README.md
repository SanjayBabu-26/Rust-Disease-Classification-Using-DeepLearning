# Rust Disease Classification Using Deep Learning Algorithm: The Case of Wheat

## Overview

This project is a Streamlit application designed to classify wheat images into different types of rust diseases using a deep learning model. The application allows users to upload images of wheat, processes them through a pre-trained deep learning model, and provides predictions regarding the presence of wheat rust diseases.

## Features

- **Wheat Rust Identification**: Classifies images into categories like "healthy", "leaf rust", "stem rust", or "Unknown".
- **Confidence Score**: Provides a confidence percentage for the classification.

## Dependencies

The project requires the following Python libraries:

- `json`
- `logging`
- `PIL` (Pillow)
- `numpy`
- `streamlit`
- `albumentations`
- `opencv-python`
- `torch`or `tensorflow` (Depending on your purpose)
- `albumentations.pytorch`

You can install the necessary libraries using pip:

```bash
pip install streamlit albumentations opencv-python torch pillow numpy
