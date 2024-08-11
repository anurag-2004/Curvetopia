# Curvetopia

#Overview  
This project focuses on developing a Convolutional Neural Network (CNN) to automatically detect and classify basic geometric shapes, including circles, squares, triangles, and ellipses, from image files. The core idea is to preprocess these images by converting them into polylines, which simplifies the shape data, making it easier for the CNN to learn and make predictions.

The workflow involves the following key steps:

Image Preprocessing: Converting .png images into vectorized polylines.
Model Training: Using these vectorized images to train a CNN model, incorporating regularization techniques to enhance model performance and generalization.
Model Testing: Testing the trained model on new, handmade images, allowing it to predict the shapes in these images.   

#Installation  
To set up the project on your local machine, follow these steps:

1.Clone the Repository:

bash   
git clone https://github.com/your-repo-url.git   
cd shape-detection-cnn   

2.Install the Required Dependencies:

bash  
pip install -r requirements.txt


Usage  
To use this project, follow these steps:

1.Vectorize the Images:

Use the vectorize_image() function to convert .png images into polylines.  
Save these vectorized images using the save_vectorized_images() function.

2.Train the Model:

Open the provided Jupyter notebook.  
Run the training cells to train the CNN on the vectorized images.  

3.Test the Model:

Upload new handmade .png images.  
Convert these images to polylines and use the model to predict the shape.  

#Project Structure  
data/:  
  raw/: Contains the original .png images of shapes.  
  vectorized/: Contains the vectorized images saved as .npy files.

models/:  
Contains the trained model file (shape_detector.h5).

notebooks/:  
Regularizing_and_detecting_shapes.ipynb: The main notebook for training and testing the model.

README.md: Provides an overview of the project.

requirements.txt: Lists all dependencies required to run the project.

#Dataset

The dataset comprises various .png images, each representing a specific shape (circle, square, triangle, or ellipse).
These images are vectorized into polylines to simplify the data and make it suitable for training the CNN.    

#Training the Model     
The CNN is trained using TensorFlow and Keras, with a focus on extracting spatial features from the vectorized shape data.
Regularization techniques like L2 regularization are employed to prevent overfitting and improve the model's ability to generalize to new data.

#Testing the Model   
After training, the model is tested on new, unseen images.
These images are also vectorized into polylines, and the model predicts the shape in the image.
The accuracy and performance of the model are evaluated using metrics like accuracy, precision, and recall.

#Results  
The model demonstrates strong performance in accurately identifying and classifying shapes from both the training data and new handmade images.
Detailed results and performance metrics can be found in the Jupyter notebook.

#License  
This project is licensed under the MIT License.


