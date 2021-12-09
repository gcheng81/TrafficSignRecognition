import cv2
import pickle
from util import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

np.random.seed(23)

def preprocess_features(X):
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    return X


def show_samples_from_generator(image_datagen, X_train, y_train):
    # take a random image from the training set
    img_rgb = X_train[0]

    # plot the original image
    plt.figure(figsize=(1, 1))
    plt.imshow(img_rgb)
    plt.title('Example of RGB image (class = {})'.format(y_train[0]))
    plt.show()

    # plot some randomly augmented images
    rows, cols = 4, 10
    fig, ax_array = plt.subplots(rows, cols)
    for ax in ax_array.ravel():
        augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
        ax.imshow(np.uint8(np.squeeze(augmented_img)))
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.suptitle('Random examples of data augmentation (starting from the previous image)')
    plt.show()


def get_image_generator():
    # create the generator to perform online data augmentation
    image_datagen = ImageDataGenerator(rotation_range=30., width_shift_range=5.0, height_shift_range=5.0)
    return image_datagen


def get_model(dropout_rate):
    input_shape = (32, 32, 1)

    input = Input(shape=input_shape)
    #******* 1st stage ********
    #cv2d_1 = Conv2D(6, (5, 5), padding='VALID', activation='relu')(input)
    #pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_1)
    cv2d_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_1)
    #dropout_1 = Dropout(dropout_rate)(pool_1)
    
    #******* 2nd stage ********
    #path2
    # flatten_1 = Flatten()(dropout_1)
    #path1
    #cv2d_2 = Conv2D(16, (5, 5), padding='VALID', activation='relu')(pool_1)
    #pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_2)
    cv2d_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_2)
    
    #******* 3rd stage ********
    #cv2d_3 = Conv2D(120, (5, 5), padding='VALID', activation='relu')(pool_2)
    cv2d_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool_2)
    #dropout_3 = Dropout(dropout_rate)(cv2d_3) # prevent overfitting
    flatten_3 = Flatten()(cv2d_3) 
    
    #concatenate feature map
    # concat_1 = Concatenate()([flatten_1, flatten_2])
    #dense_1 = Dense(84, activation='relu')(flatten_3)
    dense_1 = Dense(512, activation='relu')(flatten_3)
    dropout_1 = Dropout(dropout_rate)(dense_1)
    dense_2 = Dense(128, activation='relu')(dropout_1)
    dropout_2 = Dropout(dropout_rate)(dense_2)
    output = Dense(43, activation='softmax')(dropout_2)
    model = Model(inputs=input, outputs=output)
    # compile model
    opt = optimizers.Adam(learning_rate = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # summarize model
    model.summary()
    
    return model


def train(model, image_datagen, x_train, y_train, x_validation, y_validation):
    # checkpoint
    filepath = "./traffic-signs-data/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    print('checkpoint')
    callbacks_list = [checkpoint]
    image_datagen.fit(x_train)
    history = model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=240,
                        validation_data=(x_validation, y_validation),
                        epochs=200,
                        callbacks=callbacks_list,
                        verbose=1)

    # list all data in history
    print(history.history.keys())
# =============================================================================
#     # summarize history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
# =============================================================================
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# =============================================================================
#     with open('/trainHistoryDict.p', 'wb') as file_pi:
#         pickle.dump(history.history, file_pi)
# =============================================================================
        
    return history


def evaluate(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    accuracy = score[1]
    return accuracy


def train_model():
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("Number of training examples =", n_train)
    print("Image data shape  =", image_shape)
    print("Number of classes =", n_classes)

    X_train_norm = preprocess_features(X_train)
    y_train = utils.to_categorical(y_train, n_classes)

    # split into train and validation
    VAL_RATIO = 0.2
    X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train,
                                                                test_size=VAL_RATIO,
                                                                random_state=0)

    model = get_model(0.2)
    image_generator = get_image_generator()
    train(model, image_generator, X_train_norm, y_train, X_val_norm, y_val)


if __name__ == "__main__":
    train_model()
# =============================================================================
#     X_train_t, y_train_t = load_traffic_sign_data('./traffic-signs-data/train.p')
#     image_datagen_t = get_image_generator()
#     show_samples_from_generator(image_datagen_t, X_train_t, y_train_t)
# =============================================================================