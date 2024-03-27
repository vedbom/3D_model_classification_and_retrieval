# 3D_model_classification_and_retrieval
Neural network created using TensorFlow designed to classify images of 3D models.

The processing code (sketch_3D_model_from_OFF.pde) is used to convert a 3D model dataset into a dataset of 2D images. This is done by first rendering the 3D model in the processing environment and then taking images of the render with virtual cameras placed at different angels. The python code is used to create a convolutional neural network (CNN) that classifies the images of the 3D models into different categories. Please take a look at the attached report for more details.
