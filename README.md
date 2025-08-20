# Hand Gesture Recognition

This project implements a deep learning model to recognize hand gestures using the Leap Gesture Recognition dataset. The model is built with TensorFlow/Keras and uses a Convolutional Neural Network (CNN) architecture for accurate gesture classification.

## Dataset

The model is trained on the [Leap Gesture Recognition Dataset](https://www.kaggle.com/gti-upm/leapgestrecog) from Kaggle. The dataset contains 20,000 images of 10 different hand gestures from 10 different people.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd hand-gesture-recognition
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog)
2. Extract the dataset and place the `leapGestRecog` folder in the project root directory
3. The directory structure should look like this:
   ```
   hand-gesture-recognition/
   ├── leapGestRecog/
   │   ├── 00/
   │   ├── 01/
   │   ├── ...
   │   └── 09/
   ├── hand_gesture_recognition.py
   ├── requirements.txt
   └── README.md
   ```

## Usage

### Training the Model

To train the hand gesture recognition model, run:

```bash
python hand_gesture_recognition.py
```

This will:
1. Load and preprocess the dataset
2. Train the CNN model
3. Evaluate the model on the test set
4. Save the trained model as `hand_gesture_model.h5`
5. Generate plots showing training/validation accuracy and loss
6. Display a confusion matrix and classification report

### Model Architecture

The model uses a CNN architecture with the following layers:
- Three convolutional blocks, each with:
  - Two convolutional layers with ReLU activation
  - Batch normalization
  - Max pooling
  - Dropout for regularization
- A fully connected layer with 512 units and ReLU activation
- Output layer with softmax activation for multi-class classification

### Hyperparameters

- Image size: 64x64 pixels (grayscale)
- Batch size: 32
- Epochs: 20 (with early stopping)
- Optimizer: Adam
- Learning rate: Default (0.001) with reduction on plateau
- Loss function: Categorical cross-entropy

## Results

After training, the model will display:
- Training and validation accuracy/loss curves
- Confusion matrix
- Classification report with precision, recall, and F1-score

## Customization

You can modify the following parameters in `hand_gesture_recognition.py`:
- `IMG_SIZE`: Change the input image size
- `BATCH_SIZE`: Adjust the batch size based on your GPU memory
- `EPOCHS`: Set the maximum number of training epochs
- `NUM_CLASSES`: Number of gesture classes (default: 10)

## Real-time Gesture Recognition (Optional)

To use the trained model for real-time gesture recognition using your webcam, you can extend the code by adding a new script that:
1. Loads the trained model
2. Captures video from the webcam
3. Processes each frame to detect hands
4. Classifies the hand gesture in real-time

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Leap Gesture Recognition Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)
- TensorFlow/Keras
- OpenCV
