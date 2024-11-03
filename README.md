# Gesture Recognition for Smart TV Control

## Project Overview
This project involves building a gesture recognition system that enables users to control a smart TV without a remote. The model can identify five different gestures: thumbs up, thumbs down, left swipe, right swipe, and stop. Each gesture triggers a specific command, such as volume control or pausing the video. This gesture recognition system leverages deep learning and is implemented using two models: a 3D Convolutional Network (Conv3D) and a CNN-RNN hybrid with Gated Recurrent Units (GRUs).

## Directory Structure
The following files are provided in this project submission:
- `Gesture_Recognition_Project.ipynb`: The main Jupyter Notebook with data loading, preprocessing, model building, training, and evaluation.
- `conv3d_best_model.keras`: The best Conv3D model with architecture and weights.
- `conv3d_best_weights.weights.h5`: The best model's weights in `.h5` format.
- `Gesture_Recognition_Writeup.pdf`: Detailed report covering data processing, model architecture, experiments, and conclusions.
- `README.md`: Instructions for setup, running the code, and evaluating the model.

## Requirements
Ensure you have the following libraries installed in your environment:
- Python 3.7+
- `numpy`
- `pandas`
- `Pillow`
- `keras` and `tensorflow` (version 2.x)
- `h5py`

You can install the dependencies using:
```bash
pip install numpy pandas pillow keras tensorflow h5py
```

## Dataset
The dataset consists of short videos representing five gestures. Each video is divided into 30 frames, and each frame is stored as an image file. The dataset should be organized as follows:
- `train/`: Folder containing training videos, where each video is a subfolder with 30 images.
- `val/`: Folder containing validation videos, where each video is a subfolder with 30 images.
- `train.csv`: CSV file with training data, including the video folder name, gesture type, and label.
- `val.csv`: CSV file with validation data, including the video folder name, gesture type, and label.

## Project Structure
The project includes the following main components:

### 1. Data Preprocessing
- The `generator` function loads frames from each video folder, resizes and normalizes them, and generates batches for model training and validation.
- Preprocessing steps include resizing each frame to 64x64 pixels and normalizing pixel values to [0, 1].

### 2. Model Architectures
- **Conv3D Model**: Utilizes 3D convolutions to process spatial and temporal data, capturing gesture sequences effectively.
- **CNN-RNN Model**: Combines 2D convolutional layers for spatial feature extraction with a GRU layer to handle temporal sequences.

### 3. Training and Evaluation
- **Training**: Each model is trained using a `fit` method with `ModelCheckpoint` callbacks to save the best model and weights.
- **Evaluation**: The best saved model is loaded and evaluated on the validation set to measure accuracy and loss.

## Running the Project

### 1. Run the Jupyter Notebook
1. Open `Gesture_Recognition_Project.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure the paths to `train/`, `val/`, `train.csv`, and `val.csv` are correct.
3. Execute each cell sequentially to:
   - Load and preprocess data.
   - Define and train the Conv3D and CNN-RNN models.
   - Evaluate the model and save the best weights.

### 2. Evaluate the Model
After training, the best model is saved as `conv3d_best_model.keras`, and the weights are saved as `conv3d_best_weights.weights.h5`.

To load and evaluate the model:
1. In the notebook, load the model:
   ```python
   from keras.models import load_model

   best_model = load_model('conv3d_best_model.keras')
   val_loss, val_acc = best_model.evaluate(val_generator, steps=validation_steps)
   print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
   ```
2. This will output the validation loss and accuracy for the best model.

## Experiment Summary

The project includes several experiments that helped refine the model’s architecture and performance. Detailed information on each experiment and its results can be found in the write-up (`Gesture_Recognition_Writeup.pdf`).

## Results
The Conv3D model achieved the best results, with a validation accuracy of 83% and a validation loss of 0.43. The CNN-RNN model also performed well, reaching around 82% validation accuracy. Full experiment details are provided in the project write-up.

## References and Further Reading
- **3D Convolutional Networks**: For video processing and spatiotemporal feature extraction.
- **CNN-RNN Hybrid Models**: Useful for sequential data that requires both spatial and temporal analysis.

## Contact
For any questions or further information, please contact withawasthi@gmail.com.
