import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import os
import tensorflow as tf
import keras
from keras import layers, models
from PIL import Image
import sys

# the batch size of the dataset
batch_size = 128
# the number of neurons in the second to last dense layer, also the dimension of the descriptor vector
desc_len = 4096
# the learning rate of the CNN
learning_rate = 1e-5
# the number of epochs
num_epochs = 20

# system path that contains the dataset
img_folder_dir = r"D:\Important Documents\Processing\Processing\sketch_3D_model_from_OFF\ModelNet10Images2"
class_names = os.listdir(img_folder_dir)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-3] == class_names
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [224, 224])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# create a dataset containing the paths to the view 1 images
view_datasets = tf.data.Dataset.list_files(img_folder_dir + fr"\*\*\view_47.jpg")
img_count = len(view_datasets)

# shuffle the dataset
view_datasets = view_datasets.shuffle(buffer_size=1000)

# split the dataset into a training, validation and testing set
# skip the specified number of elements and take the rest as the training set
train_dataset = view_datasets.skip(int(img_count*0.2))
# take the specified number of elements from the top of the dataset as the validation set
val_dataset = view_datasets.take(int(img_count*0.2))
test_dataset = val_dataset.take(int(img_count*0.1))
val_dataset = val_dataset.skip(int(img_count*0.1))
# use the map method to run the process_path function on each element of the dataset
train_dataset = train_dataset.map(process_path)
val_dataset = val_dataset.map(process_path)
test_dataset = test_dataset.map(process_path)
num_train_samples = len(train_dataset)
num_val_samples = len(val_dataset)
num_test_samples = len(test_dataset)
print("Training Samples: " + str(num_test_samples))
print("Validation Samples: " + str(num_val_samples))
print("Testing Samples: " + str(num_test_samples))
# batch the datasets
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
# configure the datasets for performance
train_dataset = train_dataset.cache()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache()
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache()
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# train and test the CNN
view_in = keras.Input(shape=(224, 224, 3), name="view_in")
view_conv0 = layers.Conv2D(filters=96, kernel_size=(11, 11), activation='relu', strides=(4, 4), padding='same')(view_in)
view_batch0 = layers.BatchNormalization()(view_conv0)
view_maxpool0 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(view_batch0)
view_conv1 = layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', strides=(4, 4), padding='same')(view_maxpool0)
view_batch1 = layers.BatchNormalization()(view_conv1)
view_maxpool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(view_batch1)
view_conv2 = layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(view_maxpool1)
view_batch2 = layers.BatchNormalization()(view_conv2)
view_conv3 = layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(view_batch2)
view_batch3 = layers.BatchNormalization()(view_conv3)
view_conv4 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(view_batch3)
view_batch4 = layers.BatchNormalization()(view_conv4)
view_maxpool4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(view_batch4)
view_flatten = layers.Flatten()(view_maxpool4)
view_dense0 = layers.Dense(4096, activation='relu')(view_flatten)
view_dropout0 = layers.Dropout(0.5)(view_dense0)
view_dense1 = layers.Dense(4096, activation='relu')(view_dropout0)
view_dropout1 = layers.Dropout(0.5)(view_dense1)
view_dense2 = layers.Dense(10, activation='relu')(view_dropout1)

# create the model
CNN = keras.Model(inputs=view_in, outputs=view_dense2)
CNN.summary()

# compile the model
CNN.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# # train the model
# history = CNN.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=num_epochs
# )
# print(history.history.keys())
#
# # visualize the training and validation accuracy over the number of epochs
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.plot(range(num_epochs), history.history['sparse_categorical_accuracy'], label="Training Accuracy")
# plt.plot(range(num_epochs), history.history['val_sparse_categorical_accuracy'], label="Validation Accuracy")
# plt.legend(loc="lower right")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Accuracy")
#
# plt.subplot(1, 2, 2)
# plt.plot(range(num_epochs), history.history['loss'], label="Training Loss")
# plt.plot(range(num_epochs), history.history['val_loss'], label="Validation Loss")
# plt.legend(loc="upper right")
# plt.title("Training and Validation Loss")
# plt.xlabel("Number of Epochs")
# plt.ylabel("Loss")
# plt.show()

# save the weights of the model
#CNN.save_weights("./checkpoints/CNN")

# load the weights of the model
CNN.load_weights("./checkpoints/CNN")


# evaluate the accuracy of the model
test_loss, test_accuracy = CNN.evaluate(test_dataset)
print("Testing Accuracy: " + str(test_accuracy))

# add a softmax layer at the end of the CNN model to make the results easier to understand
CNN_softmax = tf.keras.Sequential([CNN, tf.keras.layers.Softmax()])
predictions = CNN_softmax.predict(test_dataset)
print("Predicted: ")
for i in range(10):
    print(str(np.argmax(predictions[i])))
test_images, test_labels = next(iter(test_dataset))
print("test_images shape: " + str(test_images.numpy().shape))
print("Actual: " + str(test_labels.numpy()[0:10]))


# get the output of the last fully connected layer
# get_layer method will return the layer from the keras model if given its name
print(CNN.get_layer(name="dense_1"))
print(type(CNN.get_layer(name="dense_1")))
# create a submodel from the already trained CNN in order to access the output of hidden layers
CNN_submodel = tf.keras.Model(inputs=CNN.inputs, outputs=CNN.get_layer(name="dense_1").output)
# give the test_images dataset as an input to submodel
# the submodel_output is a tensor and must be converted into a numpy array
submodel_output = CNN_submodel(test_images)
print(type(submodel_output))
print(submodel_output.numpy().shape)
print("Final layer output: " + str(submodel_output.numpy()[0]))

# feed all samples from the training dataset into the submodel and create an array of descriptors
# train_set_descriptors = np.empty((128, 224, 224, 3))
train_set_descriptors = np.empty((batch_size, desc_len))
train_set_labels = np.empty((batch_size, ))
train_set_images = np.empty((batch_size, 224, 224, 3))
num_batches = 0
for train_images_batch, train_labels_batch in iter(train_dataset):
    submodel_output = CNN_submodel(train_images_batch)
    print(submodel_output.numpy().shape)
    if num_batches == 0:
        train_set_descriptors[0:batch_size, 0:desc_len] = submodel_output.numpy()
        train_set_labels[0:batch_size] = train_labels_batch.numpy()
        train_set_images = train_images_batch.numpy()
    else:
        train_set_descriptors = np.concatenate((train_set_descriptors, submodel_output.numpy()), axis=0)
        train_set_labels = np.concatenate((train_set_labels, train_labels_batch.numpy()), axis=0)
        train_set_images = np.concatenate((train_set_images, train_images_batch.numpy()), axis=0)
    num_batches += 1
print("train_set_descriptors.shape: " + str(train_set_descriptors.shape))
print("train_set_labels.shape: " + str(train_set_labels.shape))
print("train_set_images.shape: " + str(train_set_images.shape))

# feed all samples from the testing dataset into the submodel and create an array of descriptors
test_set_descriptors = np.empty((batch_size, desc_len))
test_set_labels = np.empty((batch_size, ))
test_set_images = np.empty((batch_size, 224, 224, 3))
num_batches = 0
for test_images_batch, test_labels_batch in iter(test_dataset):
    submodel_output = CNN_submodel(test_images_batch)
    print(submodel_output.numpy().shape)
    if num_batches == 0:
        test_set_descriptors[0:batch_size, 0:desc_len] = submodel_output.numpy()
        test_set_labels[0:batch_size] = test_labels_batch.numpy()
        test_set_images[0:batch_size] = test_images_batch.numpy()
    else:
        test_set_descriptors = np.concatenate((test_set_descriptors, submodel_output.numpy()), axis=0)
        test_set_labels = np.concatenate((test_set_labels, test_labels_batch.numpy()), axis=0)
        test_set_images = np.concatenate((test_set_images, test_images_batch.numpy()), axis=0)
    num_batches += 1
print("test_set_descriptors.shape: " + str(test_set_descriptors.shape))
print("test_set_labels.shape: " + str(test_set_labels.shape))
print("test_set_images.shape: " + str(test_set_images.shape))

# print the confusion matrix for the classification task
# evaluate the model on the test set
# the prediction output will be a probability distribution so use argmax to find the largest index along the rows
test_set_predictions = np.argmax(CNN_softmax.predict(test_set_images), axis=1)
confusion_matrix = tf.math.confusion_matrix(test_set_labels, test_set_predictions).numpy()
print("Confusion Matrix: " + str(confusion_matrix))

# visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_matrix)
ax.set_xticks(np.arange(10), labels=class_names)
ax.set_yticks(np.arange(10), labels=class_names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=j, y=i, s=confusion_matrix[i, j], va="center", ha="center", size="xx-large")
plt.xlabel("Predicted", fontsize=18)
plt.ylabel("Actual", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.show()


# function will return the euclidean distance between vectors a and b
def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(np.subtract(a, b))))


# function will return the manhattan distance between vectors a and b
def manhattan_distance(a, b):
    return np.sum(np.abs(np.subtract(a, b)))


# function will return the cosine distance between vectors a and b
def cosine_distance(a, b):
    mag_a = np.sqrt(np.sum(a.dot(a)))
    mag_b = np.sqrt(np.sum(b.dot(b)))
    return 1 - a.dot(b)/(mag_a * mag_b)


# use the Euclidean distance to find the similarity measure between the descriptors in the train_set_descriptors array and the test_set_descriptors array
# similarities is a numpy array with the rows being the sample from the test set and the columns being the samples from the train set
similarities = np.empty((num_test_samples, num_train_samples))
for test_index in range(test_set_descriptors.shape[0]):
    test_descriptor = test_set_descriptors[test_index, :]
    # normalize the test_descriptor
    test_descriptor = np.divide(test_descriptor, np.sqrt(test_descriptor.dot(test_descriptor)))
    for train_index in range(train_set_descriptors.shape[0]):
        train_descriptor = train_set_descriptors[train_index, :]
        # normalize the train_descriptor
        train_descriptor = np.divide(train_descriptor, np.sqrt(train_descriptor.dot(train_descriptor)))
        euclidean_dist = euclidean_distance(train_descriptor, test_descriptor)
        manhattan_dist = manhattan_distance(train_descriptor, test_descriptor)
        cosine_dist = cosine_distance(train_descriptor, test_descriptor)
        similarities[test_index, train_index] = cosine_dist

# check if the similarities matrix has any NaN values and set them to zero
is_nan = np.isnan(similarities)
similarities[is_nan] = 0
print(similarities)

# record the positions of the top 100 values in the rows of the similarities array
# sort the similarities from highest to lowest along the rows and store the indices of the elements
sorted_indices = np.argsort(similarities, axis=1)
print(sorted_indices)
print("test sample class: " + str(test_set_labels[0]))
print("retrieved sample classes: " + str(train_set_labels[sorted_indices[0, 0:100]]))

# average the retrieval accuracy over all the testing samples
retrieval_accuracy = np.zeros((num_test_samples,))
for j in range(num_test_samples):
    for i in range(100):
        if test_set_labels[j] == train_set_labels[sorted_indices[j, i]]:
            retrieval_accuracy[j] += 1
print("retrieval accuracy: " + str(np.average(retrieval_accuracy, axis=0)))

# visualize the retrieval data for the first sample in the testing dataset
plt.figure(figsize=(10, 10))
ax = plt.subplot(4, 4, 1)
plt.imshow(test_set_images[0].astype("uint8"))
label = test_set_labels[0].astype("uint8")
plt.title("query: " + class_names[label])
plt.axis("off")
for i in range(15):
    ax = plt.subplot(4, 4, i+2)
    plt.imshow(train_set_images[sorted_indices[0, i]].astype("uint8"))
    label = train_set_labels[sorted_indices[0, i]].astype("uint8")
    plt.title(class_names[label])
    plt.axis("off")
plt.show()
