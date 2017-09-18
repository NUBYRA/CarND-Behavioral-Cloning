import cv2
import pandas as pd
import os
import matplotlib.image as mpimg
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, Flatten, Dense, Dropout, ELU
from  keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

data_dir = 'data/'
filepath = 'data/driving_log.csv'
image_height, image_width, image_depth = 64, 64, 3
batch_size = 64

def load_image(data_dir, image_path):
    return mpimg.imread(os.path.join(data_dir,image_path.strip()))

def crop_image(image):
    return image[60:-25, :, :]

def resize_image(image):
    return cv2.resize(image, (image_width, image_height))

def image_preprocess(image):
    img = crop_image(image)
    img = resize_image(img)
    return img

def random_image(data_dir, image_paths):
    correction = 0.20
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, image_paths['left']), image_paths['steering'] + correction
    elif choice == 1:
        return load_image(data_dir, image_paths['right']), image_paths['steering'] - correction
    return load_image(data_dir, image_paths['center']), image_paths['steering']   

def random_flip(image, steering):
    if np.random.rand() < .5:
        image = cv2.flip(image, 1)
        steering = -steering 
    return image, steering 

def random_translation(image, steering, range_x, range_y):
    translation_x = range_x*(np.random.uniform() - .5)
    translation_y = range_y*(np.random.uniform() - .5)
    steering = steering + translation_x*.004
    Trans_M = np.float32([[1,0,translation_x],[0,1,translation_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, Trans_M, (width, height))
    return image, steering

 def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .5 * (np.random.rand() - 0.5)
    image[:,:,2] = image[:,:,2]*random_bright 
    image[:,:,2][image[:,:,2]>255]  = 255
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def random_shadow(image):
    brightness = np.random.uniform(.2, .5)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    x = random.randint(0, image.shape[1])
    y = random.randint(0, image.shape[0])
    width = random.randint(0, image.shape[1]) * random.randint(2, 10)
    height = random.randint(0, image.shape[0]) * random.randint(2, 10)
    image[y:y+height,x:x+width,1] = image[y:y+height,x:x+width,1]*brightness
    image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    return image

def image_augmentation(data_dir, image_paths, range_x, range_y):
    image, steering  = random_image(data_dir, image_paths)
    image, steering = random_flip(image, steering)
    image, steering = random_translation(image, steering, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering

def batch_generator(data_dir, sample, batch_size, is_training):
    images = np.empty([batch_size, image_height, image_width, image_depth])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(sample.shape[0]):
            image_paths = sample.loc[index]
            steering = sample.loc[index]['steering']
            if is_training:
                image, steering = image_augmentation(data_dir, image_paths, 100, 10)
            else:
                image = load_image(data_dir, image_paths['center']) 
            images[i] = image_preprocess(image)
            steers[i] = steering
            i = i + 1
            if i == batch_size:
                break 
        yield images, steers

def create_train_valid(filepath): 
    valid_size = .2
    data = pd.read_csv(filepath)
    X_data = data[["center", "left", "right"]]
    y_data = data['steering']
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size = valid_size)
    train = X_train.assign(steering = y_train)
    train = train.reset_index()
    valid = X_valid.assign(steering = y_valid)
    valid = valid.reset_index()
    return train, valid

def COMMA():
    '''COMMA.ai model'''
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape = (image_height, image_width, image_depth)))
    model.add(Conv2D(16, (8, 8), strides=(4,4), activation='elu', padding = 'same'))
    model.add(Conv2D(32, (5, 5), strides=(2,2), activation='elu', padding = 'same'))
    model.add(Conv2D(64, (5, 5), strides=(2,2), activation='elu', padding = 'same'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.summary()
    return model

def train_model(data_dir, filepath, batch_size):
    train, valid = create_train_valid(filepath)
    model = COMMA()
    checkpoint = ModelCheckpoint('model3.h5', monitor = 'val_loss', verbose = 0, save_best_only = True, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1.0e-4))
    train_generator = batch_generator(data_dir, train, batch_size, True)
    validation_generator = batch_generator(data_dir, valid, batch_size, False)
    model.fit_generator(train_generator, 
                    steps_per_epoch = 20000, 
                    validation_data = validation_generator,
                    validation_steps = len(valid), 
                    epochs = 3,
                    callbacks =[checkpoint])

def main():
	train_model(data_dir, filepath, batch_size)

if __name__ == "__main__":
	main()