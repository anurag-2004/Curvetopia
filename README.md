
# Curvetopia

# Regularizing and Detecting Shapes

This project focuses on training and testing a Convolutional Neural Network (CNN) model to identify and detect various shapes such as circles, squares, triangles, and ellipses. The model is trained using `.png` image files that are converted into polylines, vectorized, and regularized before training. The trained model can then predict shapes from new images, even if they are hand-drawn.

## Project Structure

- **Training**: The model is trained using `.png` images converted into polylines. The images are vectorized and regularized before being fed into the CNN for training.
- **Testing**: The trained model is tested on handmade images that are uploaded as `.png` files. These images are also converted into polylines and vectorized before the model predicts whether they are circles, squares, triangles, or ellipses.

## Installation

To run this project, you need to have the following Python libraries installed:

- `cv2` (OpenCV)
- `numpy`
- `tensorflow`
- `matplotlib`
- `os`
- `sklearn`

You can install the necessary packages using pip:

```bash
pip install opencv-python numpy tensorflow matplotlib scikit-learn
```


# Usage
## Vectorizing Images

The project begins by vectorizing the .png images to prepare them for model training and testing:  
 Convert to grayscale  
 Apply binary thresholding  
 Find contours  
 Create an empty image to draw contours  
 Draw contours as polylines  
 Save the resulting image

## Training the Model
The CNN model is built and trained on the vectorized images:  
 Define input shape  
 Build CNN model  
 Train the CNN model  
 Evaluate the model

## Testing the Model
After training, the model can predict shapes from new images:  
 Read the image 
 Vectorize the input image 
 Reshape for the model 
 Make predictions 
 Interpret predictions 
 Visualize the image with contours and prediction 

## Project Structure
Regularizing_and_Detecting_Shapes.ipynb: The main notebook containing the code and explanation for predicting various shapes.

## Contributing
Feel free to submit issues or pull requests if you have suggestions for improvements or new features.

# Finding the Possible Symmetries for the Regularized Shapes

This project aims to identify possible symmetries in regularized shapes using various computational techniques. The notebook provides methods to process images, vectorize contours, and detect symmetries, displaying results graphically.

## Installation

To run the notebook, you need to have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `os`
- `cv2` (OpenCV)

You can install the necessary packages using pip:

```bash
pip install numpy matplotlib opencv-python
```
## Usage
The notebook provides an example workflow for detecting symmetries in shapes:

1.Create a figure with two subplots: one for the image and one for symmetry results.  
2.Plot the image with contours.  
3.Detect symmetries and display the results.

## Example Usage 
 1.Example code to detect and visualize symmetries  
 2.Create a figure with two subplots: one for the image and one for symmetry results  
 3.Plot the image with contours  
 4.Plot the symmetry results  
 5.Flatten the contour to ensure correct shape  
 6.Flatten the rotated contour to compare  
 7.Vectorize the image  
 8.Detect symmetries  
 9.Draw and display the vectorized contours with symmetry results

## Project Structure
Finding_Possible_Symmetries.ipynb: The main notebook containing the code and explanation for detecting symmetries.

## Contributing
Feel free to submit issues or pull requests if you have suggestions for improvements or new features.
