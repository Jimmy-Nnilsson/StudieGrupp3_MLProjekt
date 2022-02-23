import os
from pyexpat import model
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
print("TF: ", tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import models
from keras import Model
import keras

import seaborn as sns

from keras.preprocessing.image import load_img, img_to_array, image_dataset_from_directory
from tensorflow.keras.applications import vgg19
from sklearn.metrics import confusion_matrix, classification_report

import cv2
from keras.applications.vgg16 import VGG16
    
class Model_Class():
    def __init__(self,
                 model_path):

        self.model = self.__get_model(model_path)
        
        self.conv_layers, self.layer_names = self.__get_convlayers()
        self.preds = ""

    def __get_model(self, path):
        kwarg = dict(include_top=False, input_shape=(224,224,3))
        self.base_model = vgg19.VGG19(**kwarg)
        # self.base_model = vgg16.VGG16(**kwarg)
        # self.base_model = mobilenet_v2.MobileNetV2(**kwarg)
        # self.base_model = inception_v3.InceptionV3(**kwarg)

        self.base_model.trainable = False

        x = layers.GlobalAveragePooling2D()(self.base_model.output)

        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model  = models.Model(self.base_model.input, outputs)
        model.load_weights(path)
        return model

    def load_model(self, path):
        model = VGG16(weights = path)
        return model

    def grad_cam(self, image, layer=None):
        self.preds = self.predict(image)

        if type(layer) is list:
            heatmap_list, superimposed_list = {},{}
            for layer_num in layer:
                heatmap = self.make_gradcam_heatmap(np.expand_dims(image, axis=0), layer_num, np.argmax(self.preds[0]))
                superimposed_img = self.superimpose(image,heatmap)
                heatmap_list[self.model.layers[layer_num]._name] = heatmap
                superimposed_list[self.model.layers[layer_num]._name] = superimposed_img
            return image, heatmap_list, superimposed_list
        else:
            heatmap = self.make_gradcam_heatmap(np.expand_dims(image, axis=0), layer, np.argmax(self.preds[0]))
            superimposed_img = self.superimpose(image,heatmap)

            heatmap = cv2.resize(heatmap, dsize=(224,224))
            return image, heatmap, superimposed_img


    def make_gradcam_heatmap(self, img_array, layer=None, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        if layer == None: layer=self.conv_layers[-1]
        model = self.model
        grad_model = Model(
            [model.inputs], [model.layers[layer].output, model.output]
            # [model.inputs], [model.get_layer(self.layer_names[-1]).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def predict(self, pic):
        x = model_preprocess(pic)
        if len(x.shape) < 4:
          x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x)
        return preds

    def superimpose(self, pic,heatmap):
        img_numpy = np.asarray(pic)

        heatmap_resized = cv2.resize(heatmap, (img_numpy.shape[1], img_numpy.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        superimposed_img = 0.3*heatmap_resized[:,:,::-1] + img_numpy
        superimposed_img = superimposed_img.astype(np.uint8)
        return superimposed_img

    def __get_convlayers(self):
        list_conv_layers = []
        list_layer_names = []
        for i,l in enumerate(self.model.layers):
            # print(str(l).split('.'))
            if str(l).split('.')[2] == 'convolutional':
                list_conv_layers.append(i)
                list_layer_names.append(l._name)
        return list_conv_layers, list_layer_names

    def set_trainable(self, trainable_layers=0): # Sets whole model to trainable
        if trainable_layers == 0:
            self.model.trainable = True
        else:
            self.model.trainable = False
            for i in trainable_layers:
                self.model.layers[i].trainable = True


def crcm(model, x, y):
    y_pred = model.predict(x)
    y_pred = np.asarray(y_pred)
    y_pred = np.uint8(y_pred+0.5)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred)
    plt.xlabel('Pred')
    sns.heatmap(cm, vmin=0, annot=True)
    print(cr)

def model_preprocess(images):
    images = vgg19.preprocess_input(images)
    # images = vgg16.preprocess_input(images)
    # images = mobilenet_v2.preprocess_input(images)
    # images = inception_v3.preprocess_input(images)
    return images