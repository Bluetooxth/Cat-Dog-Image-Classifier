# Cats vs Dogs CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using TensorFlow and Keras. The model is deployed using Gradio for easy interaction.

## Project Structure
```
.
├── app.py                      # Gradio web interface
├── main.py                     # Training script
├── cats_vs_dogs_classifier.h5  # Trained model
├── requirements.txt            # Project dependencies
├── train/                     # Training data images
│   ├── cats/
│   └── dogs/
└── validation/                # Validation data
```

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- keras
- gradio
- numpy
- opencv-python

## Training the Model

1. Prepare your dataset in the `train` and `validation` directories
2. Run the training script:
```bash
python3 main.py
```

## Using the Web Interface

To start the Gradio interface:
```bash
python3 app.py
```

The interface will be available at `http://localhost:7860`

## Model Architecture

The CNN model uses a sequential architecture with:
- Convolutional layers
- Max pooling
- Dropout for regularization
- Dense layers for classification

## Dataset

Place your training images in:
- `train/cats/` for cat images
- `train/dogs/` for dog images

Similarly for validation data.