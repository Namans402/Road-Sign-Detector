#%% Libraries/Initial Set Up 

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import cv2

#%% Reuseable Functions

class_lookup = {0: 'Speed Limit 20',
                1: 'Speed Limit 30',
                2: 'Speed Limit 50',
                3: 'Speed Limit 60',
                4: 'Speed Limit 70',
                5: 'Speed Limit 80',
                6: 'End of Speed Limit 80',
                7: 'Speed Limit 100',
                8: 'Speed Limit 120',
                9: 'No Passing',
                10: 'No Passing (by vehicles over 3.5t)',
                11: 'Priority to through-traffic',
                12: 'Priority Road',
                13: 'Yield',
                14: 'Stop',
                15: 'No vehicles permitted',
                16: 'No Lkw (Trucks) permitted',
                17: 'Do Not Enter',
                18: 'Danger',
                19: 'Dangerous curve to the left',
                20: 'Dangerous curve to the right',
                21: 'Double curves',
                22: 'Bumpy road',
                23: 'Slippery Road',
                24: 'Road narrows on the right',
                25: 'Construction Areas',
                26: 'Traffic Signals',
                27: 'Pedestrians',
                28: 'Children',
                29: 'Cyclists',
                30: 'Slippery due to snow/ice',
                31: 'Wild animals',
                32: 'End of Previous Limitations',
                33: 'Must turn right ahead',
                34: 'Must turn left ahead',
                35: 'Must go straight ahead',
                36: 'Must go straight or turn right',
                37: 'Must go straight or turn left',
                38: 'Keep right of divider',
                39: 'Keep left of divider',
                40: 'Roundabout',
                41: 'End of No Passing',
                42: 'End of No Heavy Vehicle Passing'}

def show_pred(test_data, true, pred, idx):
    img = array_to_img(test_data[idx])
    true_argmax = np.argmax(true, axis = 1)
    pred_argmax = np.argmax(pred, axis = 1)
    plt.title(f'True: {class_lookup.get(true_argmax[idx])}\nPred: {class_lookup.get(pred_argmax[idx])}')
    plt.imshow(img)
    plt.show()

# preset the resolution of every image
image_size = 150

# based on sample value, go into all the classes and retrive an even amount of images from each class along with their lables.
# images retirived are randomly selected in a a way where we don't get duplicates
def load_train(path, samples):
    
    dataset = []
    labels = np.array([], dtype=np.int16)
    path="archive/Train"
    # if samples = -1, get all images    
    if samples == -1:
        for label in tqdm(range(0, 43), desc='Load Train Classes'):
            for root, dirs, files in os.walk(f'{os.getcwd()}/{path}/{label}'):
                for file_name in files:
                    dataset.append(img_to_array(load_img(f'{path}/{label}/{file_name}', target_size=(image_size, image_size))))
                    labels = np.append(labels, label)
    else:
        samples_each = int((samples/43)+1)
        for label in tqdm(range(0, 43), desc='Load Train Classes'):
            for root, dirs, files in os.walk(f'{os.getcwd()}/{path}/{label}'):
                # if a class has less images compared to samples, get all images for that class
                if samples_each >= len(files):
                    for file_name in files:
                        dataset.append(img_to_array(load_img(f'{path}/{label}/{file_name}', target_size=(image_size, image_size))))
                        labels = np.append(labels, label)
                # get an evenly distributed aoumnt of images  
                else:
                    for i in random.sample(range(0, len(files)), int((samples/43)+1)):
                        dataset.append(img_to_array(load_img(f'{path}/{label}/{files[i]}', target_size=(image_size, image_size))))
                        labels = np.append(labels, label)
        
    dataset = np.array(dataset)
            
    return dataset, to_categorical(labels)

# load all test images
def load_test():
    
    image = []
    labels = []
    all_labels = pd.read_csv("archive/Test.csv").ClassId.to_numpy()

    for i in tqdm(range(0, 12630), desc = 'Load test images'):
        image.append(img_to_array(load_img(f'archive/Test/{str(i).zfill(5)}.png', target_size=(image_size, image_size))))
        labels.append(all_labels[i])
    return np.array(image), to_categorical(labels)

# CNN model model architecture
def CNN():
    n_filter = 40
    classifier = Sequential()
    
    classifier.add(Conv2D(n_filter, (11, 11), input_shape = (image_size, image_size, 3), activation = 'relu'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
        
    classifier.add(Conv2D(n_filter, (7, 7), activation = 'relu'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
       
    classifier.add(Conv2D(n_filter, (3, 3), activation = 'relu'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(5000, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(1000, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(500, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(43, activation = 'softmax'))
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

    classifier.summary()
    
    return classifier    

# plots
def plots(history, title):
    # Plot the accuracy for both train and validation set
    plt.subplots() # open a new plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(f'{title} model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    
    # Plot the loss for both train and validation set
    plt.subplots() # open a new plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    
def imageGen(data, label, augments=0):
    
    datagen = ImageDataGenerator(
            #rotation_range=60,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.3,
            zoom_range=0.3,
            # horizontal_flip=True,
            # vertical_flip = True,
            fill_mode='nearest'
            )
    
    if augments < len(data):
        aug_data = datagen.flow(data[:augments], label[:augments], batch_size=augments)
    elif augments == len(data):
        aug_data = datagen.flow(data, label, batch_size=augments)
    else:
        temp_data = []
        temp_label = []
        # Append the whole data for (augments // len(data)) times
        for i in range(augments // len(data)):
            temp_data.append(data)
            temp_label.append(label)
        # Randomly add the remainders to temp_data
        temp_data = np.array(temp_data)
        temp_data = temp_data.reshape(len(temp_data[0]), image_size, image_size, 3)
        temp_label = np.array(temp_label)
        temp_label = temp_label.reshape(len(temp_label[0]), 43)
        for i in random.sample(range(0, len(data)), augments % len(data)):
            temp_data = np.append(temp_data, [data[i]], axis=0)
            temp_label = np.append(temp_label, [label[i]], axis=0)
            
            # Get aug_data from temp_data
            aug_data = datagen.flow(np.array(temp_data), np.array(temp_label), batch_size=augments)
    
    return aug_data.__getitem__(0)

def showImage(images, amounts):
    for i in range(amounts):
          # get image called image
          cv2.imshow(f'Augmented Image {i}' , images[i])
          if cv2.waitKey(0) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
    
#%% Part 1) 200 training image CNN

samples = 200

# Import train data
train_data, train_label = load_train("archive/Train", samples)

# Import test data
test_data, test_label = load_test()

# Fit pixel intensity range between 0 and 1
train_data = train_data/255.0
test_data = test_data/255.0

# CNN model
classifier = CNN()

# creates image of model architecture
# plot_model(classifier, to_file='200t_model.png', show_shapes=True)

# save the best epoch
checkpointer = ModelCheckpoint(filepath="200t_best_weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = classifier.fit(
                         train_data, 
                         train_label, 
                         epochs=30, 
                         batch_size=30,
                         #validation_split = 0.2, 
                         callbacks = [checkpointer], 
                         validation_data = (test_data, test_label)
                         )

# save the whole model incase we take breaks in between training
#model_name = "200t_cnn.h5"
#classifier.save(model_name)
#classifier = load_model(model_name)

y_pred = classifier.predict(test_data)
y_pred_argmax = np.argmax(y_pred, axis = 1)

cm = confusion_matrix((np.argmax(y_pred, axis = 1)), (np.argmax(test_label, axis = 1)))

print("\nHistory Keys:\n")
print(history.history.keys())

# Plot the accuracy and loss for both train and validation set
plots(history, "200t")

# Show predicted label for model
show_pred(test_data, test_label, y_pred, 3)
#%% Part 2.1) Create 300 augmented images from 200 training images 

# number of augmented images we want
augments = 300

aug_data, aug_label = imageGen(train_data, train_label, augments=augments)

# showImage(aug_data, 5)

#%% Part 2.2) 200 training and 300 augmented images CNN

# CNN model
classifier = CNN()

# save the best epoch
checkpointer = ModelCheckpoint(filepath=f"{samples}t{augments}a_best_weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = classifier.fit(
                         np.append(train_data, aug_data, axis=0), 
                         np.append(train_label, aug_label, axis=0), 
                         epochs=30, 
                         batch_size=30,
                         #validation_split = 0.2, 
                         callbacks = [checkpointer], 
                         validation_data = (test_data, test_label)
                         )

# save the whole model incase we take breaks in between training
#model_name = "200t_cnn.h5"
#classifier.save(model_name)
#classifier = load_model(model_name)

y_pred = classifier.predict(test_data)
y_pred_argmax = np.argmax(y_pred, axis = 1)

cm = confusion_matrix((np.argmax(y_pred, axis = 1)), (np.argmax(test_label, axis = 1)))

print("\nHistory Keys:\n")
print(history.history.keys())

# Plot the accuracy and loss for both train and validation set
plots(history, f"{samples}t{augments}a")

# Show predicted label for model
show_pred(test_data, test_label, y_pred, 3)

#%% Part 3) 1000 training image CNN

samples = 1000

# Import train data
train_data, train_label = load_train("archive/Train", samples)

# Import test data
test_data, test_label = load_test()

# Fit pixel intensity range between 0 and 1
train_data = train_data/255.0
test_data = test_data/255.0

# CNN model
classifier = CNN()

# creates image of model architecture
# plot_model(classifier, to_file='1000t_model.png', show_shapes=True)

# save the best epoch
checkpointer = ModelCheckpoint(filepath="1000t_best_weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = classifier.fit(
                         train_data, 
                         train_label, 
                         epochs=30, 
                         batch_size=30,
                         #validation_split = 0.2, 
                         callbacks = [checkpointer], 
                         validation_data = (test_data, test_label)
                         )

# save the whole model incase we take breaks in between training
#model_name = "1000t_cnn.h5"
#classifier.save(model_name)
#classifier = load_model(model_name)

y_pred = classifier.predict(test_data)

cm = confusion_matrix((np.argmax(y_pred, axis = 1)), (np.argmax(test_label, axis = 1)))

print("\nHistory Keys:\n")
print(history.history.keys())

# Plot the accuracy and loss for both train and validation set
plots(history, "1000t")

# Show predicted label for model
show_pred(test_data, test_label, y_pred, 12000)

#%% Part 4.1) Create 1000 augmented images from 1000 training images 

# number of augmented images we want
augments = 1000

aug_data, aug_label = imageGen(train_data, train_label, augments=augments)

# showImage(aug_data, 5)

#%% Part 4.2) 1000 training and 1000 augmented images CNN

# CNN model
classifier = CNN()

# save the best epoch
checkpointer = ModelCheckpoint(filepath=f"{samples}t{augments}a_best_weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

train_aug_data = np.append(train_data, aug_data, axis=0)
train_aug_label = np.append(train_label, aug_label, axis=0)

history = classifier.fit(
                         train_aug_data, 
                         train_aug_label, 
                         epochs=30, 
                         batch_size=30,
                         #validation_split = 0.2, 
                         callbacks = [checkpointer], 
                         validation_data = (test_data, test_label)
                         )

# save the whole model incase we take breaks in between training
#model_name = "200t_cnn.h5"
#classifier.save(model_name)
#classifier = load_model(model_name)

y_pred = classifier.predict(test_data)
y_pred_argmax = np.argmax(y_pred, axis = 1)

cm = confusion_matrix((np.argmax(y_pred, axis = 1)), (np.argmax(test_label, axis = 1)))

print("\nHistory Keys:\n")
print(history.history.keys())

# Plot the accuracy and loss for both train and validation set
plots(history, f"{samples}t{augments}a")

# Show predicted label for model
show_pred(test_data, test_label, y_pred, 12000)

#%% Part 5) CNN using all data except augmented images

samples = -1

# Import train data
train_data, train_label = load_train("archive/Train", samples)

# Import test data
test_data, test_label = load_test()

# Fit pixel intensity range between 0 and 1
train_data = train_data/255.0
test_data = test_data/255.0

# CNN model
classifier = CNN()

# creates image of model architecture
# plot_model(classifier, to_file='-1t_model.png', show_shapes=True)

# save the best epoch
checkpointer = ModelCheckpoint(filepath="-1t_best_weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = classifier.fit(
                         train_data, 
                         train_label, 
                         epochs=10, 
                         batch_size=30,
                         #validation_split = 0.2, 
                         callbacks = [checkpointer], 
                         validation_data = (test_data, test_label)
                         )

# save the whole model incase we take breaks in between training
#model_name = "-1t_cnn.h5"
#classifier.save(model_name)
#classifier = load_model(model_name)

classifier = load_model("-1t_best_weights.hdf5")

y_pred = classifier.predict(test_data)

cm = confusion_matrix((np.argmax(y_pred, axis = 1)), (np.argmax(test_label, axis = 1)))

print("\nHistory Keys:\n")
print(history.history.keys())

# Plot the accuracy and loss for both train and validation set
plots(history, '-1t')

# Show predicted label for model
show_pred(test_data, test_label, y_pred, 216)
