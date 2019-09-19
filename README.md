This repo creates a custom image classifier with Keras, and it then compares that classifier's performance with two other classifiers. The other classifiers are based on the ResNet50 and InceptionV3 architectures, with new dense layers added to them.

The dataset being used is the Flowers Recognition dataset (available [HERE](https://www.kaggle.com/alxmamaev/flowers-recognition)).

There is a Jupyter notebook attached to this repo and the notebook demonstrates how to implement a custom classifier architecture, as well as how to use pre-defined architectures. The notebook will also allow users to adjust arguments of the models like training epochs and batch size to see how they impact the performance of the classifier.

Initially, we import all the libraries we need.

```Python
import random as rn
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Flatten, Activation
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
```

Next, we create a function to load in the data. (Thanks to [Raj Mehrotra](https://www.kaggle.com/rajmehra03) at Kaggle for the layout of the function for loading the data.)

```Python
# separating training data into different folders

images = []
labels = []

img_size = 150

def training_data(im_class, dir):
    # tqdm is for the display of progress bars
    # for the image in the listed directory
    for img in tqdm(os.listdir(dir)):
        label = im_class
        path = os.path.join(dir, img)
        _, ftype = os.path.splitext(path)
        if ftype == ".jpg":
            # read in respective image as a color image
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            # rsize the respective image
            img = cv2.resize(img, (img_size, img_size))

            # make a numpy array out of the image
            images.append(np.array(img))
            labels.append(str(label))

training_data('Daisy', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/daisy')
print(len(images))

training_data('Sunflower', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/sunflower')
print(len(images))

training_data('Tulip', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/tulip')
print(len(images))

training_data('Dandelion', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/dandelion')
print(len(images))

training_data('Rose', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/rose')
print(len(images))
```

Next, the labels are encoded and made categorical/one-hot, while the features are turned into arrays. Train/test-split is then used to create the training and testing features and labels. Some random seeds are also set.

```Python
# do label encoding
encoder = LabelEncoder()
y_labels = encoder.fit_transform(labels)
# one hot encoding
y_labels = to_categorical(y_labels, 5)
x_features = np.array(images)
# data normalization
x_features = x_features/255

X_train, X_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.25, random_state=27)

np.random.seed(27)
rn.seed(27)
# graph level random seed for TF/Keras
tf.set_random_seed(27)
```

Some of the data can be visualized to ensure proper loading.

```Python
# will make 5 x 2 subplots
fig, ax = plt.subplots(2, 5)
# set a specific size for the plot
fig.set_size_inches(15, 15)
# specify plotting on both axis
for i in range(2):
    for j in range(5):
        # get a random integer between one and the final label in the list of labels
        label = rn.randint(0, len(labels))
        ax[i, j].imshow(images[label])
        ax[i, j].set_title('Flower: '+labels[label])
plt.tight_layout()
plt.show()
```

We now create an image data generator to apply random perturbations to the training data.

```Python
# because the model is so large/deep it may be a good idea to perform some data augmentation in order
# combat overfitting
data_generator = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2,
                                    horizontal_flip=True)

# fit on the trainign data to produce more data with the specified transformations
data_generator.fit(X_train)
```

This section creates the model.

```Python
def create_model():
    # first specify the sequential nature of the model
    model = Sequential()
    # conv2d is the convolutional layer for 2d images
    # first parameter is the number of memory cells - let's just try 64 units for now
    # second parameter is the size of the "window" you want the CNN to use
    # the shape of the data we are passing in, 3 x 150 x 150
    model.add(Conv2D(64, (5, 5), input_shape=(150, 150, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),  padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # flatten the data for the dense layer
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    # now compile the model, specify loss, optimization, etc
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
```

We'll declare our batch size and number of epochs to use for the first round of training and then fit the model.

```Python
batch_size = 64
num_epochs = 50

# fit the generator, since we used one in making new data

filepath = "C:/Users/Daniel/Downloads/flowers-recognition/weights_custom.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
             EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

train_records = model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs = num_epochs,
                          validation_data=(X_test, y_test), verbose= 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=callbacks)
```

We'll create some functions to visualize the accuracy and loss and evaluate the model's predictions.

```Python
# visualize training loss
# declare important variables
training_acc = train_records.history['acc']
val_acc = train_records.history['val_acc']
training_loss = train_records.history['loss']
validation_loss = train_records.history['val_loss']

# gets the lengt of how long the model was trained for
train_length = range(1, len(training_acc) + 1)

def plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss):

    # plot the loss across the number of epochs
    plt.figure()
    plt.plot(train_length, training_loss, label='Training Loss')
    plt.plot(train_length, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(train_length, training_acc, label='Training Accuracy')
    plt.plot(train_length, val_acc, label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def make_preds(model):
    # make predictions
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\nAchieved Accuracy:', score[1],'\n')

    y_pred = model.predict(X_test)
    # evalute model predictions
    Y_pred_classes = np.argmax(y_pred, axis=1)
    Y_true = np.argmax(y_test, axis=1)
    confusion = confusion_matrix(Y_true, Y_pred_classes)
    print(confusion)

plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss)
make_preds(model)
```

![](C:\jupyter_notebooks\Flowers Classification\training_acc_1.png)

![](C:\jupyter_notebooks\Flowers Classification\training_loss_1.png)

Here's the results of the predictions function:

```Python
Achieved Accuracy: 0.7835337644258547

[[151  21  19   9   4]
 [ 19 229   5  19   6]
 [  4   6 137   3  14]
 [  2  11   6 173   7]
 [  5  15  49  10 157]]
```

The rest of the program trains the two other models: Resnet50 and Inception.

Take note that for the purposes of this demonstration the custom classifier is trained on the dataset for longer than the two fine-tuned models. This is to demonstrate that, in the case of the ResNet based model, the model approaches the performance of the custom model in only 10 epochs of training. Meanwhile, the Inception based model trains for 10 epochs and receives only about 20% accuracy. This shows how  using the correct pre-defined architecture is important.

Here's the model instantiation and training setup for the ResNet model.

```Python
# Create the base model and add it to our sequential framework
resnet = ResNet50(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))
print(resnet.summary())
resnet_model = Sequential()
resnet_model.add(resnet)

# Add in our own densely connected layers (after flattening the inputs)
resnet_model.add(Flatten())
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dropout(0.2))
resnet_model.add(Dense(64, activation='relu'))
resnet_model.add(Dropout(0.2))
resnet_model.add(Dense(5, activation='softmax'))
resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(resnet_model.summary())

num_epochs = 10

filepath = "C:/Users/Daniel/Downloads/flowers-recognition/weights_resnet50.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
             EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

resnet_records = resnet_model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs = num_epochs,
                          validation_data=(X_test, y_test), verbose= 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=callbacks)
```

Here's the evaluation for the ResNet model.

```Python
# visualize training loss
# declare important variables
training_acc = resnet_records.history['acc']
val_acc = resnet_records.history['val_acc']
training_loss = resnet_records.history['loss']
validation_loss = resnet_records.history['val_loss']

# gets the length of how long the model was trained for
train_length = range(1, len(training_acc) + 1)

plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss)
make_preds(resnet_model)
```

![](C:\jupyter_notebooks\Flowers Classification\training_loss_2.png)



![](C:\jupyter_notebooks\Flowers Classification\training_acc_2.png)

Here's the results of the predictions analysis.

```Python
Achieved Accuracy: 0.7169287690512016 

[[134  25  20   7  18]
 [  9 214  15  24  16]
 [  6  13  91   0  54]
 [  5  17   6 158  13]
 [  3  12  34   9 178]]
```

Finally, here's the setup and training for the Inception model.

```Python
Incep = InceptionV3(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))
Incep_model = Sequential()
Incep_model.add(Incep)
Incep_model.add(Flatten())
Incep_model.add(Dense(256, activation='relu'))
Incep_model.add(Dropout(0.2))
Incep_model.add(Dense(64, activation='relu'))
Incep_model.add(Dropout(0.2))
Incep_model.add(Dense(5, activation='softmax'))
print(Incep_model.summary())
Incep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = "C:/Users/Daniel/Downloads/flowers-recognition/weights_Incep19.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
             EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

num_epochs = 10

Incep_records = Incep_model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs = num_epochs,
                          validation_data=(X_test, y_test), verbose= 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=callbacks)
```

Here's the evaluation.

```Python
# visualize training loss
# declare important variables
training_acc = Incep_records.history['acc']
val_acc = Incep_records.history['val_acc']
training_loss = Incep_records.history['loss']
validation_loss = Incep_records.history['val_loss']

# gets the length of how long the model was trained for
train_length = range(1, len(training_acc) + 1)

plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss)
make_preds(Incep_model)
```

![](C:\jupyter_notebooks\Flowers Classification\training_loss_3.png)



![](C:\jupyter_notebooks\Flowers Classification\training_acc_3.png)

```Python
Achieved Accuracy: 0.20444033305254608

[[ 37   0   0   0 167]
 [ 71   0   0   0 207]
 [ 21   1   0   0 142]
 [ 39   5   0   5 150]
 [ 51   5   0   1 179]]

```

