import os
# from pyexpat import model
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
import streamlit as st

import seaborn as sns

from keras.preprocessing.image import load_img, img_to_array, image_dataset_from_directory
from tensorflow.keras.applications import vgg19
from sklearn.metrics import confusion_matrix, classification_report

import cv2
from keras.applications.vgg19 import VGG19

import numpy as np
import cv2
import pathlib

class Model_Class():
    def __init__(self,
                 model_path):

        # self.model = self.__get_model(model_path)
        self.model = self.load_model(model_path)

        self.conv_layers, self.layer_names = self.__get_convlayers()
        self.preds = ""

    def __get_model(self, path):
        kwarg = dict(include_top=False, input_shape=(224,224,3))
        self.base_model = vgg19.VGG19(**kwarg)

        self.base_model.trainable = False

        x = layers.GlobalAveragePooling2D()(self.base_model.output)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        model  = models.Model(self.base_model.input, outputs)
        model.load_weights(path)
        return model

    def load_model(self, path):
        self.model = keras.models.load_model(path)
        return self.model

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
        pic = cv2.resize(pic, dsize=(224,224))
        x = model_preprocess(pic)
        if len(x.shape) < 4:
          x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x)
        return preds

    def superimpose(self, pic,heatmap):
        img_numpy = np.asarray(np.clip(pic, 0, 190))

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

    # def set_trainable(self, trainable_layers=0): # Sets whole model to trainable
    #     if trainable_layers == 0:
    #         self.model.trainable = True
    #     else:
    #         self.model.trainable = False
    #         for i in trainable_layers:
    #             self.model.layers[i].trainable = True


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

class BEX_cropping():
    """Cropping class som tar emot fullständig sökväg, cropppar enligt algoritm o returnerar np-array"""

    def __init__(self,f_name=None):
        self.file_name = f_name


    def calculate_cropping(self, img):
        frame = 3                                             # klipper av en ram runt bilden direkt eftersom mkt smuts sitter där
        IMG_SIZE = (224, 224)                                    # detta är storleken som bilderna rezas till
        shape_comparison = []
        if self.file_name != None:
            fname = self.file_name
            image = cv2.imread(str(fname))
        else:
            image = img

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        temp = gray[frame:-frame, frame:-frame]             # temp är bild-numpyn
        res_std_x = np.std(temp, axis = 0)                  #skapar en arr med standardavvikelse för x och y
        res_std_y = np.std(temp, axis = 1)

        res_std_x = self.encode(res_std_x, 0.1)                  # encode gör en arr "binär"; allt < 0.1 -> 0 allt annat = 1
        res_std_y = self.encode(res_std_y, 0.1)

        calc_arr = self.encode(temp, 40, 255)                    # ändrar tröskelvärdet i bilden. Allt under 40 -> 0, allt annat = 255

        res_sum_x = np.sum(calc_arr, axis = 0)              # skapar en arr med summor av linjer i x och y
        res_sum_y = np.sum(calc_arr, axis = 1)

        res_sum_ = self.encode(res_sum_x, 1000)                  # gör binär kodning; allt < 1000 -> 0, allt annat =1
        res_sum_y = self.encode(res_sum_y, 1000)

        sigma_x = res_std_x * res_sum_x                     # mutliplicerar ihop de båda metoderna sum * std
        sigma_y = res_std_y * res_sum_y                     # då framträder det 0 på alla rader med imformation som ska bort

        half_width = int(.5 * len(sigma_x))                 # delar bildens bredd för sökning
        half_height = int(.5 * len(sigma_y))                # delar bildens höjd för sökning

        left_val = np.where(sigma_x[:half_width] == 0)[0]       # söker från vänster mot mitten
        right_val = np.where(sigma_x[half_width:] == 0)[0]      # söker från mitten till högerkant

        top_val = np.where(sigma_y[:half_height] == 0)[0]       # söker från toppen
        bottom_val = np.where(sigma_y[half_height:] == 0)[0]    # söker från mitten ner mot botten


        if len(left_val) > 0:                       # om värde '0' hittats så är längden > 0
            left_x = left_val[-1]                   # tar sista värdet i arr dvs det närmst mitten
        else:
            left_x = 0 

        if len(right_val) > 0:                      # om värde '0' hittats så är längden > 0
            right_x = right_val[0] + half_width     # tar först värdet i arr eftersom sökningen börjar mitt i bild
        else:
            right_x = len(sigma_x)

        if len(top_val) > 0:                        # om värde '0' hittats så är längden > 0
            top_y = top_val[-1]                     # tar sista värdet i arr dvs det närmst mitten
        else:
            top_y = 0 

        if len(bottom_val) > 0:                     # om värde '0' hittats så är längden > 0
            bottom_y = bottom_val[0] + half_height  # tar först värdet i arr eftersom sökningen börjar mitt i bild
        else:
            bottom_y = len(sigma_y)

        new_image = gray[top_y + frame:bottom_y + frame, left_x + frame:right_x + frame]    # här appliceras frame på alla mått
        
        #----sparar undan shapes pre / post
        temp_shape = []
        temp_shape.append(image.shape[0:2])         # 0-2 för att skippa sista dim som bara anger lager i bild (RGB=3)
        temp_shape.append(new_image.shape)
        shape_comparison.append(temp_shape)         # sparar undanstorleken för att jämföra vilka bilde som beskärdes mest
        # -----------------------------------


        new_image = cv2.resize(new_image, IMG_SIZE)
        new_image = np.stack((new_image,)*3, axis=-1)
        return new_image

#---------------hjälp funktion--------------------------------
    def encode(self, arr, thresh, max = 1):
        arr = np.where(arr < thresh, 0, arr)
        arr = np.where(arr != 0, max, arr).astype(int)
        return arr


