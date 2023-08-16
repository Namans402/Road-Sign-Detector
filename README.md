# Road Sign Detection

### Problem Statement:

For this project, our goal is to create a machine learning classification model which classifies images. The model is to be a convolutional neural network (CNN) that will be optimized based on model accuracy. Based on these requirements, we will make a classifier for common road signs.

### Dataset:

The dataset we used was the famous German Traffic Sign Recognition Benchmark (GTSRB) dataset that could be found on [Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) or the [official site](https://benchmark.ini.rub.de/gtsrb_dataset.html). The dataset contains 43 classes with 39.2 thousand training images and 12.6 thousand testing images. Image size varies between 15x15 and 250x250 pixels and are not necessarily always square images. The signs in the images themselves are not always centered in the frame of the image. While the dataset does come with annotations in the form of an excel file for both the train and test data, this will be ignored for the purposes of this project. It is important to remember that the dataset is not balanced as shown in the [image below](https://medium.com/@thomastracey/recognizing-traffic-signs-with-cnns-23a4ac66f7a7).

![image](https://user-images.githubusercontent.com/32663193/126677042-3d672350-74fa-4eb1-9736-9180904d463f.png)

[3] Graph 1: Number of Samples per Class

### Model:

Before starting on the model there were a few issues we had to consider. First, we needed to load an even amount of train images from each class, so our model is exposed to each class proportionally. This is important when looking at confusion matrices later to narrow down issues with our model as we do not want a poor preforming model to stem from non-diverse data. This also prevents a feature dominating over other features. Second, we wanted to have 5 test cases where we used 200, 1000 and all train images as well as mixture of 200 train 300 augmented images and 1000 train 1000 augmented images. This is so that we have an abundance of data at our disposal, as the more high-quality data we have the better our model will learn all the different classes.

For the first problem we wrote a custom function called load_train that requested the number of images the user wanted. If they wanted all the training images (denoted as -1), if not the function would divide the number of requested images with the number of total classes to calculate the number of images requested per class. From here we check the number of images each class has. If the class did not contain enough images, we simply return all the images in the class. If the class contained enough images, we randomly picked the required number of images from the class. While extracting each image, we define the resolution of each image to deal with the varying image sizes.

```
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
```

For the second problem, we created a custom function called imageGen that would create augmented images. We used the ImageDataGenerator function from Keras to produce the augmented images based on parameters we set such as zoom, sheer, width and height shifts, rotation, etc.

```
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
```

With all the main issues sorted out and the code created, it was time to create the CNN. Our final CNN included 3 sets of 2D convolution + 2D average pooling layers where the filter size dropped from 11x11 to 7x7 to 3x3, a flatten layer, 4 dense layers the first 2 dense layers were accompanied by a dropout layer which would drop 20 percent of the current neurons in each dense layer.

```
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
```
 
### Results:

Our first CNN was tested on the 200-training data. It had 3 sets of 2D convolution + 2D average pooling layers where the filter size dropped from 11x11 to 7x7 to 3x3. The ideal for this was to capture general characteristics in the beginning and then start to look at more specific details later. After the 3 sets on convolution and pooling layers, we added a flatten and 7 dense layers. There were 7 dense layers to slowly reduce the 9000 neurons to 43 neurons at a slow rate to prevent any major loss of data however, we noticed that the performance of this model was unideal. Our best epoch had a validation accuracy of 0.2561.

```
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
    
    classifier.add(Dense(6300, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(3087, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(1510, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(510, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(250, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(100, activation = 'relu', kernel_initializer='uniform'))
    classifier.add(Dense(43, activation = 'softmax'))
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

    classifier.summary()
    
    return classifier 
```

| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126680232-253c38a1-7a42-4b30-99c6-c750a9c28839.png) | ![image](https://user-images.githubusercontent.com/32663193/126680253-cf495305-0554-44f0-a070-a49d2dad65bf.png) 

Figure 1: First CNN architecture with 200 training images

From here we reduced the number of dense layers in the CNN as well as adding dropout layers and normalizing the pixel intensities. This seemed to increase the model performance on the 200-image dataset and so we decided to benchmark this new model, let us call it version 2, with all the other datasets.


| Dataset |	Dataset code |	Validation Accuracy |
|---------|--------------|----------------------|
| For 200 training images: |	200t |	Best val acc = 0.4143 |
| For 200 training and 300 augmented images: |	200t 300a |	Best val acc = 0.2213 |
| For 1000 training images: |	1000t |	Best val acc = 0.7382 |
| For 1000 training and 1000 augmented images: |	1000t 1000a |	Best val acc = 0.5972 |
| For all training images: |	-1t |	Best val acc = 0.9527 |

```
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
```

<br>

| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126680353-0f2661be-dbc6-4cdd-a33d-cfcec58d4b88.png) | ![image](https://user-images.githubusercontent.com/32663193/126680377-2e434714-5cd1-48f0-b183-704a0b019fe9.png) 

Figure 2: Second CNN architecture with 200 training images

| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126680410-5b399ba3-53f1-4c0e-8510-551295c9b057.png) | ![image](https://user-images.githubusercontent.com/32663193/126680425-6b4acca9-9009-4792-bb31-ddf086ff78c3.png) 

Figure 3: Second CNN architecture with 200 training and 300 augmented images

| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126690865-fc03b023-a2a8-49b6-8c3a-01b5609d8dae.png) | ![image](https://user-images.githubusercontent.com/32663193/126691277-c28b9abf-a169-4b30-bc01-db036f5a0551.png)
 
 	 
Figure 4: Second CNN architecture with 1000 training images
| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126691312-5b579391-1d18-4d0c-8b7a-5ca1b9b4d160.png) | ![image](https://user-images.githubusercontent.com/32663193/126691333-057e4334-74ea-4666-a8e9-8cb7d1e08fad.png) 
 	 
Figure 5: Second CNN architecture with 1000 training and 1000 augmented images

| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126691365-561995a6-6191-4f20-860d-af2e6042a7a2.png) | ![image](https://user-images.githubusercontent.com/32663193/126691384-db7eb0d6-9b41-4075-884b-55197c173b53.png) 
 	 
Figure 6: Second CNN architecture with all training images

Being satisfied with the second CNN model, we started looking into optimizing the model by changing the following listed below and comparing them to the base models above. Outside of the second CNN architecture, it uses 40 filers per Conv2D, sheer and rotation and an image size of 150x150. All the models run on 30 epochs except for -1t which runs on 10 epochs.

| Changes |	Dataset code |	Validation Accuracy |
|---------|--------------|----------------------|
| No sheer, no rotation |	200t 300a |	Best val acc = 0.2373 |
| Sheer but no rotation |	200t 300a |	Best val acc = 0.2594 |
| No sheer but rotation |	200t 300a |	Best val acc = 0.2318 |
| Non-normalized pixels |	1000t |	Best val acc = 0.0594 |
| Image size of 100x100 |	1000t |	Best val acc = 0.8143 |
| Image size of 200x200 |	1000t |	Best val acc = 0.6509 |
| 20 filters per Conv2D |	1000t |	Best val acc = 0.7392 |
| 60 filters per Conv2D |	1000t |	Best val acc = 0.7546 |
| Max pooling |	1000t |	Best val acc = 0.7080 |
| Sigmoid activation layers |	1000t |	Best val acc = 0.0594 |

Looking at all these test cases, it seems like sheer, no rotation, normalized pixel intensities, smaller image size, and more filters will give us the best model yet. Let us test that model on 1000t 1000a and see how it goes. This model will have 60 filters, an image size of 100x100, sheer and no rotation, normalized pixel intensities, and the rest will be the same as the second CNN architecture.

| Accuracy | Loss |
|----------|------|
| ![image](https://user-images.githubusercontent.com/32663193/126691758-c8ef7f3b-867e-4d1e-b5fc-1669e80b996e.png) | ![image](https://user-images.githubusercontent.com/32663193/126691787-28c1e399-a69f-4576-b6b8-4c16f1ae335d.png) 

Figure 7: Third CNN architecture with 1000 training and 1000 augmented images

The Third CNN model seems to be horrible with an accuracy of 0.0523. We are not too sure where we went wrong here but we will use our second CNN model and our final model.
