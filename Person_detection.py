# TensorFlow and tf.keras
import tensorflow as tf

# Set the number of CPUs to be used
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

from tensorflow import keras
# from tensorflow.keras import backend as K

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #  for Normalization of dataset - ELIJA
from sklearn.utils import shuffle

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd
import urllib.request
import copy

# Import training and validation dataset
import trainingdata as tdata
import validationdata as vdata

# Import tqdm for status bar during run of programm
from tqdm import tqdm
import os

# Load the TensorBoard notebook extension
#%load_ext tensorboard
import datetime

# Argument Parsing
import argparse

# Import logging
import sys
import logging

# Turn modifications on/off
IMAGE_STANDARDIZATION = 0
MULTIPLE_METRICS = 0
SOFTMAX_ACTIVATION = 1
CATEGORICAL_CROSSENTROPY = 1
ADDED_LAYERS = 0


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M', filename='output.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
console.setFormatter(formatter)
training_logger = logging.getLogger('Training')
training_logger.addHandler(console)


# Set Constants
FILENAME = 'RAISE_all.csv'
RAISE_DIR = '/original/RAISE/'#
RAISE_TRAIN_DIR = RAISE_DIR + 'RaiseTrain/'
RAISE_VAL_DIR = RAISE_DIR + 'RaiseVal/'
# MODEL_PATH = '/data/projekt1/tan_branch/models/'#
# CHECKPOINT_PATH = '/data/projekt1/tan_branch/checkpoints/cp.ckpt'

IMG_WIDTH = 536
IMG_HEIGHT = 356
BUFFER_SIZE = 2000

#init bias = ln(pos/neg). For ex: ln(460/2192)
INIT_BIAS = np.log([1/1])
output_bias = tf.keras.initializers.Constant(INIT_BIAS)

# GLOBAL VALUES
global LEARNING_RATE #= 0.000001
global BATCH_SIZE #= 10
global TOTAL_IMAGES #= 150
global EPOCHS #= 10
global NEW_MODEL #= 1
global LOGDIR
global LOAD_DIR
LOGDIR = '/tmp/'
KERNEL_REGULARIZER = 0

# # Build image feature extract model from Inception V3
# It's posible to just use image_model instead of image_features_extract_model! -- Maybe ask what the differenc is betwee those two.. - ELIJA
# image_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(IMG_HEIGHT,IMG_WIDTH,3), classes=2, pooling=None, weights='imagenet')
image_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(IMG_HEIGHT,IMG_WIDTH,3), pooling=None, weights='imagenet')
image_model.trainable = False # set weights to be untrainable/unmodifyable - ELIJA

# # tf variables
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

image_flatten = tf.keras.layers.Flatten()
if SOFTMAX_ACTIVATION == True:
    image_dense2 = tf.keras.layers.Dense(2, bias_initializer=output_bias, activation='softmax', kernel_regularizer=keras.regularizers.l2(l=KERNEL_REGULARIZER)) ## added softmax activation from this answer: https://stackoverflow.com/questions/59410176/keras-why-binary-classification-isnt-as-accurate-as-categorical-calssification -- ELIJA
else:
    image_dense2 = tf.keras.layers.Dense(2, bias_initializer=output_bias, kernel_regularizer=keras.regularizers.l2(l=KERNEL_REGULARIZER))

# Image augmentation layer for the model
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    # tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    # tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
])

opt = tf.keras.optimizers.Adam(learning_rate=10e-6)

# Function to help convert possible strings which are meant to be bool into real boolean values
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def format_sample(image, label):
#     image = tf.cast(image, tf.float32)
#     image = tf.keras.applications.inception_v3.preprocess_input(image)
#     return image, label

# Get all images that contain people in it
def get_pos_dataset(img_vector, class_vector):
    pos_img_vector = []
    pos_class_vector = []
    for i in range(len(class_vector)):
        if class_vector[i] == [1, 0]:
            pos_class_vector.append(class_vector[i])
            pos_img_vector.append(img_vector[i])
    return pos_img_vector, pos_class_vector

# Get all images that do not contain people in it
def get_neg_dataset(img_vector, class_vector):
    neg_img_vector = []
    neg_class_vector = []
    for i in range(len(class_vector)):
        if class_vector[i] == [0, 1]:
            neg_class_vector.append(class_vector[i])
            neg_img_vector.append(img_vector[i])
    return neg_img_vector, neg_class_vector

def preprocess_raise_img_vector(all_img_path_vector):
    all_img_vector = []
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    # rescale=1. / 255,
    #shear_range=30,
    #zoom_range=30,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #rotation_range=20,
    horizontal_flip=True)

    
    # for index in tqdm(range(len(all_img_path_vector))):
    for index in range(len(all_img_path_vector)):

        img_path = all_img_path_vector[index]
        # Save image as numpy array in vector
        img = image.imread(img_path)
        img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = tf.cast(img, tf.float32)
        # img = tf.image.per_image_standardization(img) # Standardize img to have mean 0 and variance 1
        # training_logger.info(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        if(IMAGE_STANDARDIZATION == True):
            img = train_datagen.standardize(img)
        # training_logger.info(img)

        all_img_vector.append(img)

    return all_img_vector

# def preprocess_raise_db_img(all_img_path_vector):
#     # Store image names and image paths in vectors
#     all_img_vector = []

#     raise_db = pd.read_csv(RAISE_DIR + FILENAME)

#     image_paths = raise_db.File 

#     counter = 0    
#     for row in tqdm(range(image_paths.shape[0])):
#         if counter == TOTAL_IMAGES:
#             break
#         if counter == 4195:
#             continue
#         if counter == 4196:
#             continue
#         if counter == 4197:
#             continue
#         if counter == 4198:
#             continue
        
#         counter = counter+1

#         # Save image as numpy array in vector
#         img = image.imread(all_img_path_vector[row])  
#         img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
#         img = tf.keras.applications.inception_v3.preprocess_input(img)
#         img = np.expand_dims(img, axis=0)
#         img_np = np.asarray(img)

#         all_img_vector.append(img_np)

#     return all_img_vector

# def preprocess_raise_db_binary():
#     # Store image names and image paths in vectors
#     all_img_binary_vector = []

#     raise_db = pd.read_csv(RAISE_DIR + FILENAME)

#     keywords = raise_db.Keywords
#     image_paths = raise_db.File 

#     counter = 0    
#     for row in tqdm(range(image_paths.shape[0])):
#         if counter == TOTAL_IMAGES:
#             break
#         if counter == 4195:
#             continue
#         if counter == 4196:
#             continue
#         if counter == 4197:
#             continue
#         if counter == 4198:
#             continue
        
#         counter = counter+1

#         # Create binary vector
#         if('people' in str(keywords.loc[row]) ):
#             all_img_binary_vector.append([1, 0])
#             #all_img_binary_vector.append(1)
#         else:
#             all_img_binary_vector.append([0, 1])
#             #all_img_binary_vector.append(0)

#     return all_img_binary_vector

# def preprocess_raise_db_path():
#     # Store image names and image paths in vectors
#     all_img_path_vector = []
#     raise_db = pd.read_csv(RAISE_DIR + FILENAME)
#     image_paths = raise_db.File 

#     counter = 0    
#     for row in tqdm(range(image_paths.shape[0])):
#         if counter == TOTAL_IMAGES:
#             break
#         if counter == 4195:
#             continue
#         if counter == 4196:
#             continue
#         if counter == 4197:
#             continue
#         if counter == 4198:
#             continue
        
#         counter = counter+1

#         for root, dirs, files in os.walk(RAISE_DIR): 
#             for file in files:  
#                 if file.endswith(str(image_paths.loc[row]) + '.TIF'): 
#                     all_img_path_vector.append(root+'/'+str(file))

#     return all_img_path_vector

# def preprocess_raise_db():
#     all_img_path_vector = []
#     all_img_binary_vector = []

#     raise_db = pd.read_csv(RAISE_DIR + FILENAME)

#     keywords = raise_db.Keywords
#     image_paths = raise_db.File 

#     counter = 0
#     for row in tqdm(range(image_paths.shape[0])):
#         if counter == TOTAL_IMAGES:
#             break
#         if counter == 4195:
#             continue
#         if counter == 4196:
#             continue
#         if counter == 4197:
#             continue
#         if counter == 4198:
#             continue
        
#         counter = counter+1

#         # Store image names and image paths in vectors        
#         img_path = RAISE_DIR + 'RaiseTrain/' + image_paths[row] + '.TIF'
#         if os.path.isfile(img_path):
#             all_img_path_vector.append(img_path)
            
#             # Create binary vector
#             if('people' in str(keywords.loc[row]) ):
#                 all_img_binary_vector.append([1, 0])
#             else:
#                 all_img_binary_vector.append([0, 1])
    
#     return [all_img_path_vector, all_img_binary_vector]

# Count pos and neg (images with and without people respectively) in class_vector
def class_vector_count(class_vector):
    positives = 0
    for i in range(len(class_vector)):
        if class_vector[i] == [1,0]:
            positives += 1
    negatives = len(class_vector) - positives
    if negatives == 0:
        sys.exit(0)

    # # Print (im)balance information about dataset - ELIJA
    total = negatives + positives
    training_logger.info("Examples: ")
    training_logger.info("Total: ")
    training_logger.info(total)
    training_logger.info("Positive: ")
    training_logger.info(positives)
    training_logger.info("Percentage of total: ")
    perc_of_total = 100 * positives / total
    training_logger.info(perc_of_total)


    return positives, negatives

# # Evaluate
def draw_graph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return

def initialize_model():
    # # Build the main model
    if(ADDED_LAYERS == True):        
        model = tf.keras.Sequential([
            # data_augmentation,
            # image_model,
            image_features_extract_model,
            tf.keras.layers.BatchNormalization(),
            #tf.keras.Input(shape = (None, 8, 8, 2048), batch_size = 10),
            image_flatten,
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            #tf.keras.layers.Dense(2048, activation='relu', input_shape=(None, None, None, 2048)),
            #tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=keras.regularizers.l2(l=KERNEL_REGULARIZER)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(l=KERNEL_REGULARIZER)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            image_dense2, #, activation='softmax') # , activation='sigmoid', input_dim=3
        ])
    else:
        model = tf.keras.Sequential([
            # data_augmentation,
            # image_model,
            image_features_extract_model,
            #tf.keras.Input(shape = (None, 8, 8, 2048), batch_size = 10),
            image_flatten,
            #tf.keras.layers.Dense(2048, activation='relu', input_shape=(None, None, None, 2048)),
            #tf.keras.layers.Dropout(.2),
            image_dense2#, activation='softmax') # , activation='sigmoid', input_dim=3
        ])
        

    # # Print output shapes after each layer
    #for layer in model.layers:
    #    print(layer.output_shape)

    # # Compile model    
    #model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    
    # # Implementation of multiple Metrics - ELIJA
    METRICS = [
        tf.keras.metrics.categorical_accuracy,
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        #tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        #tf.keras.metrics.Precision(name='precision'),
        #tf.keras.metrics.Recall(name='recall'),
        #tf.keras.metrics.AUC(name='auc'),
    ]

    
    
    if MULTIPLE_METRICS == True and CATEGORICAL_CROSSENTROPY == True:
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=METRICS) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    if MULTIPLE_METRICS == True and CATEGORICAL_CROSSENTROPY == False:
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    if MULTIPLE_METRICS == False and CATEGORICAL_CROSSENTROPY == True:    
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['CategoricalAccuracy']) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    if MULTIPLE_METRICS == False and CATEGORICAL_CROSSENTROPY == False:
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy']) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
         
    # # Print summary
    training_logger.info(model.summary())

    return model



# Function for loading existing models
def load_model(model_dir = LOGDIR, checkpoint_dir = LOGDIR, model_name = 'Person_detection_1'):
    # # Load existing model
    # model = tf.keras.models.load_model(MODEL_PATH + model_name, compile=False)
    model = tf.keras.models.load_model(model_dir, compile=False)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    #Prevent pre-trained model from trainable
    model.layers[0].trainable = False

    # # Compile & build model    
    #model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # # Implementation of multiple Metrics - ELIJA
    METRICS = [
        tf.keras.metrics.categorical_accuracy,
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        #tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        #tf.keras.metrics.Precision(name='precision'),
        #tf.keras.metrics.Recall(name='recall'),
        #tf.keras.metrics.AUC(name='auc'),
    ]

    if MULTIPLE_METRICS == True and CATEGORICAL_CROSSENTROPY == True:
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=METRICS) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    if MULTIPLE_METRICS == True and CATEGORICAL_CROSSENTROPY == False:
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    if MULTIPLE_METRICS == False and CATEGORICAL_CROSSENTROPY == True:    
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['CategoricalAccuracy']) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    if MULTIPLE_METRICS == False and CATEGORICAL_CROSSENTROPY == False:
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy']) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
            
    #model.build(input_shape=(BATCH_SIZE,IMG_LENGTH,IMG_WIDTH,3))

    # # Print summary
    training_logger.info(model.summary())

    return model

def save_model(model, model_name='Person_detection_1'):
    # # Save model
    # model.save(MODEL_PATH + model_name)
    # training_logger.info('Model saved as ' + model_name + ' in directory ' + MODEL_PATH)
    # model.save(LOGDIR + model_name)
    model.save(LOGDIR)
    #training_logger.info('Model saved as ' + model_name + ' in directory ' + LOGDIR)
    return

# def train_and_checkpoint(manager):
#     ckpt.restore(manager.latest_checkpoint)
#     if manager.latest_checkpoint:
#         print("Restored from {}".format(manager.latest_checkpoint))
#     else:
#         print("Initializing from scratch.")

def get_weights(model):
    # training_logger.info(model.trainable_variables)
    # trainable_variables = tf.Variable(model.trainable_variables)
    # model_weights = trainable_variables.eval()
    model_weights = np.asarray(model.trainable_variables[0].numpy())
    model_weights = model_weights[:,0]
    # training_logger.info('Layer weight and output: ')
    # training_logger.info(model_weights)

    return model_weights

# Calculate false predictions
def calc_pred_stat(labels, predictions):
    total_fn = 0 # False negatives
    total_fp = 0 # False positives
    total_tn = 0 # True negatives
    total_tp = 0 # True positives
    for i in range(len(labels)):
        if (labels[i] == [1,0]):
            total_tp += 1
            if (predictions[i][0] < predictions[i][1]): #meaning if label is people and prediction is non-people
                total_fn += 1
        if (labels[i] == [0,1]):
            total_tn += 1
            if (predictions[i][0] > predictions[i][1]): #meaning if label is non-people and prediction is people
                total_fp += 1
    percentage_fn_tp = total_fn/total_tp # Percentage of FN / TP
    percentage_fp_tp = total_fp/total_tp # Percentage of FP / TP
    return {'fn': total_fn, 'fp': total_fp, 'tn': total_tn, 'tp': total_tp, 'p_fn_tp': percentage_fn_tp, 'p_fp_tp': percentage_fp_tp}

# Print false negatives and false positives into logfile
def out_fn_fp(model, dataset):
    predictions = model.predict(dataset, batch_size=BATCH_SIZE, verbose=0)
    # training_logger.info(predictions)
    class_vector = []
    for x, y in dataset.unbatch():
        y = y.numpy().tolist()
        class_vector.append(y)
    # training_logger.info(class_vector)
    pred_stat = calc_pred_stat(class_vector, predictions)
    training_logger.info('False Negatives / True Positives: ' + repr(pred_stat['fn']) + ' / ' + repr(pred_stat['tp']) + ' = ' + repr(pred_stat['p_fn_tp']))
    training_logger.info('False Positives / True Positives: ' + repr(pred_stat['fp']) + ' / ' + repr(pred_stat['tp']) + ' = ' + repr(pred_stat['p_fp_tp']))

# Print lowest loss and corresponding accuracy into logfile
def out_best_loss_and_acc(loss, acc):
    loss = np.asarray(loss)
    acc = np.asarray(acc)
    
    # lowest_loss = np.amin(loss)
    # indices_of_all_min_loss = np.where(loss == lowest_loss)
    # index_value = indices_of_all_min_loss[0][0]
    # training_logger.info("Lowest loss: "+repr(lowest_loss))
    # training_logger.info("Corresponding accuracy: "+repr(acc[index_value]))

    best_acc = np.amax(acc)
    indices_of_all_max_acc = np.where(acc == best_acc)
    index_value = indices_of_all_max_acc[0][0]
    training_logger.info("Best accuracy: "+repr(best_acc))
    training_logger.info("Corresponding loss: "+repr(loss[index_value]))


if (__name__ == "__main__"):
    #sys.stdout = open("output_log.txt", "w")

    PARSER = argparse.ArgumentParser()

    # Adding arguments for parser
    PARSER.add_argument('--logdir', type=str, default='/tmp/', help='Logfile path')
    PARSER.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    PARSER.add_argument('--batch_size', type=int, default=8, help='Defining a batchsize')
    PARSER.add_argument('--learning_rate', type=float, default=0.000001, help='Defining a learning rate')
    PARSER.add_argument('--total_images', type=int, default=80, help='Defining size of loaded dataset. Up to 8000 images of the RAISE dataset can be chosen')
    PARSER.add_argument('--kernel_regularizer', type=float, default=0, help='Defining a kernel weight regulization')
    PARSER.add_argument('--new_model', type=str2bool, default=False, help='Whether or not to create a new model (or to load an existing one)')
    PARSER.add_argument('--load_dir', type=str, default='', help='If new_model is False, defining a dir path to load saved model and checkpoint from.')
    PARSER.add_argument('--imba_sol', type=int, default=0, help='Solution for imbalanced dataset. Choose between 0: oversampling, and 1: class weighting.')
    PARSER.add_argument('--user', type=str, default='unknown', help='Whether user is Tan or Elija, in order to save results in correct folder.')

    args = PARSER.parse_args()

    #UNPARSED = PARSER.parse_known_args()
    #print('%r' % UNPARSED)
    #tf.app.run(main=main, argv=sys.argv[0] + UNPARSED)

     # Load parsed arguments and update global variables
    #LOGDIR = args.logdir

    #print(kwargs._get_kwargs)
    #print(sys.argv[0])

    for name, value in args._get_kwargs():
        variable_name = name.upper()
        exec(variable_name + " = value")
        # if name=='learningrate':
        #     LEARNING_RATE=value
        #     continue
        # if name=='batchsize':
        #     BATCH_SIZE=value
        #     continue
        # if name=='totalimages':
        #     TOTAL_IMAGES=value
        #     continue
        # if name=='epochs':
        #     EPOCHS=value
        #     continue
        # if name=='newmodel':
        #     NEW_MODEL=value
        #     continue
        # if name=='logdir':
        #     LOGDIR=value
        #     continue
        # if name=='loaddir':
        #     LOAD_DIR=value
        #     continue

    # # Get variables
    # path_vector = preprocess_raise_db_path()
    # all_img_name_vector = preprocess_raise_db_img(path_vector)
    # class_vector = preprocess_raise_db_binary()

    # # Import from trainingdata.py and validationdata.py
    train_path_vector = tdata.path_vector[:TOTAL_IMAGES]
    train_class_vector = tdata.class_vector[:TOTAL_IMAGES]
    train_all_img_name_vector = preprocess_raise_img_vector(train_path_vector)
    training_logger.info('Size of training sample: ' + str(len(train_path_vector)))
    # training_logger.info(train_all_img_name_vector.shape)
    # training_logger.info(len(train_class_vector))

    # Examine class label imbalance
    pos, neg = class_vector_count(train_class_vector)

    # Oversampling validation dataset to reduce the imbalance
    val_path_vector = vdata.path_vector[:TOTAL_IMAGES]
    val_class_vector = vdata.class_vector[:TOTAL_IMAGES]
    val_all_img_name_vector = preprocess_raise_img_vector(val_path_vector)
    # training_logger.info(val_all_img_name_vector)
    training_logger.info('Validation dataset before oversampling:')
    training_logger.info('Size of validation sample: ' + str(len(val_path_vector)))

    val_path_vector_pos = []
    val_class_vector_pos = []
    val_img_vector_pos = []
    val_pos = 0
    for i in range(len(val_class_vector)):
        if val_class_vector[i] == [1,0]:
            val_pos += 1
            val_path_vector_pos.append(val_path_vector[i])
            val_class_vector_pos.append(val_class_vector[i])
            val_img_vector_pos.append(val_all_img_name_vector[i])

    training_logger.info("Positive: " + repr(val_pos))
    training_logger.info("Percentage of total: " + repr(val_pos/len(val_class_vector)))

    val_neg = len(val_class_vector) - val_pos
    ratio = int(val_neg / val_pos) - 1
    val_path_vector_pos = [item for item in val_path_vector_pos for i in range(ratio)]
    val_class_vector_pos = [item for item in val_class_vector_pos for i in range(ratio)]
    val_img_vector_pos = [item for item in val_img_vector_pos for i in range(ratio)]
    # training_logger.info(val_path_vector_pos)
    # training_logger.info(val_class_vector_pos)
    # training_logger.info(val_img_vector_pos)

    val_path_vector += val_path_vector_pos
    val_class_vector += val_class_vector_pos
    val_all_img_name_vector += val_img_vector_pos 

    # # Train the model
    # Create training and validation sets using an 80-20 split
    # img_name_train, img_name_val, class_train, class_val = train_test_split(all_img_name_vector, class_vector, test_size=0.2, random_state=0,shuffle=True)
    img_name_train = train_all_img_name_vector
    class_train = train_class_vector
    img_name_val = val_all_img_name_vector
    class_val = val_class_vector
    # training_logger.info('Class train:')
    # training_logger.info(class_train)
    # training_logger.info('Class val:')
    # training_logger.info(class_val)
    training_logger.info('Size of img_name_train: ' + str(len(img_name_train)))
    training_logger.info('Size of img_name_val: ' + str(len(img_name_val)))


    # # Added Normalization of dataset - ELIJA -- substituted by codeline in function "preprocess_raise_img_vector"
    #scaler = StandardScaler()
    #train_dataset = scaler.fit_transform(train_dataset)
    #val_dataset = scaler.transform(val_dataset)

    #train_dataset = np.clip(train_dataset, -5, 5)
    #val_dataset = np.clip(val_dataset, -5, 5)

    #training_logger.info('Training dataset shape: ') 
    #training_logger.info(train_dataset.shape)
    #training_logger.info('Validation dataset shape: ')
    #training_logger.info(val_dataset.shape)


    # for history variable, class train and class val have to also be numpy arrays
    # class_train = np.array(class_train).astype('float32').reshape((-1,1))
    # class_val = np.array(class_val).astype('float32').reshape((-1,1))
    # class_train = np.array(class_train).astype('float32')
    # class_val = np.array(class_val).astype('float32')
    # img_name_train = np.array(img_name_train)
    # img_name_val = np.array(img_name_val)
  
    # Create tensor datasets in order to feed them into model.fit() function
    train_dataset = tf.data.Dataset.from_tensor_slices((img_name_train,class_train))
    original_val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val,class_val))

    # Set batch_size
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    val_dataset = original_val_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    # # Implement Standardization of images - Elija
    #train_dataset = tf.image.per_image_standardization(train_dataset)
    #val_dataset = tf.image.per_image_standardization(val_dataset)

    # Data augmentation for dataset
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(2)

    # Inspect:
    # for image_batch, label_batch in train_dataset.take(1):
    #     pass
    # training_logger.info(image_batch.shape)

    #img_name_train = img_name_train.reshape(10, 299, 299, 3)
    #img_name_val = img_name_val.reshape(10, 299, 299, 3)

    # Class weighting: Calculate the class weight
    if IMBA_SOL == 1:  
        true_bias = np.log([pos/neg])
        output_bias = tf.keras.initializers.Constant(true_bias)
        if SOFTMAX_ACTIVATION == True:
            image_dense2 = tf.keras.layers.Dense(2, bias_initializer=output_bias, activation='softmax', kernel_regularizer=keras.regularizers.l2(l=KERNEL_REGULARIZER)) ## added softmax activation from this answer: https://stackoverflow.com/questions/59410176/keras-why-binary-classification-isnt-as-accurate-as-categorical-calssification -- ELIJA
        else:
            image_dense2 = tf.keras.layers.Dense(2, bias_initializer=output_bias, kernel_regularizer=keras.regularizers.l2(l=KERNEL_REGULARIZER))  
        neg_weight = (1 / neg)*(TOTAL_IMAGES)/2.0 
        pos_weight = (1 / pos)*(TOTAL_IMAGES)/2.0
        class_weight = {0: pos_weight, 1: neg_weight}

    # Oversampling: oversampling pos samples to make more balance dataset
    # set_repeat = int(neg/pos)
    if IMBA_SOL == 0:
        # Create tensor datasets in order to feed them into model.fit() function
        pos_img_name_train, pos_class_train = get_pos_dataset(img_name_train, class_train)
        neg_img_name_train, neg_class_train = get_neg_dataset(img_name_train, class_train)
        pos_train_ds =  tf.data.Dataset.from_tensor_slices((pos_img_name_train,pos_class_train)).shuffle(BUFFER_SIZE).repeat()
        neg_train_ds =  tf.data.Dataset.from_tensor_slices((neg_img_name_train,neg_class_train)).shuffle(BUFFER_SIZE).repeat()
        resampled_train_ds = tf.data.experimental.sample_from_datasets([pos_train_ds, neg_train_ds], weights=[0.5, 0.5])

        # Set batch_size
        resampled_train_ds = resampled_train_ds.batch(BATCH_SIZE)
        # Data augmentation for resampled dataset
        resampled_train_ds = resampled_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        resampled_train_ds = resampled_train_ds.prefetch(2)
        resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)



    # Set the checkpoint path
    checkpoint_path = LOGDIR + 'cpp1.ckpt'

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # # Load existing or initialize new Model
    if(NEW_MODEL == True):
        training_logger.info('Value of NEW_MODEL: ' + str(NEW_MODEL))
        model = initialize_model()
        model.save_weights(checkpoint_path.format(epoch=0))
    else:
        training_logger.info("Loading model from: " + repr(LOAD_DIR))
        model = load_model(LOAD_DIR, LOAD_DIR)    

    # # Checkpoint callback
    # checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    ckpt = tf.train.Checkpoint(optimizer = opt, model = model)
    manager = tf.train.CheckpointManager(ckpt, directory="/tmp/train", max_to_keep=5)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=CHECKPOINT_PATH,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     save_best_only=True,
    #     mode='min')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1)

    # For Tensorboard Implementation -- probably unnecessary rn - ELIJA
    #log_dir = LOGDIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR, histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', embeddings_freq=0)

    # Early Stopping: stop when the val_loss not improving after 10 epochs
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=10, mode='min', restore_best_weights=False)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=40, mode='min', restore_best_weights=False)

    # Reducing learning rate when metrics stop improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=10, verbose=0, mode='max')
    
    #history = model.fit(img_name_train, class_train, shuffle=True, batch_size=10, epochs=10, validation_data=(img_name_val, class_val), callbacks=[tensorboard_callback])
    #history = model.fit(img_name_train, class_train, shuffle=True, epochs=EPOCHS, validation_data=val_dataset, callbacks=[cp_callback])
    if IMBA_SOL == 0:
        history = model.fit(resampled_train_ds, shuffle=True, epochs=EPOCHS, steps_per_epoch=resampled_steps_per_epoch, verbose=2, validation_data=val_dataset, callbacks=[tensorboard_callback, cp_callback, reduce_lr, early_stopping])
    if IMBA_SOL == 1:
        history = model.fit(train_dataset, shuffle=True, epochs=EPOCHS, verbose=2, validation_data=val_dataset, class_weight=class_weight, callbacks=[tensorboard_callback, cp_callback, reduce_lr, early_stopping])

    # # Save model after training
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    save_model(model)

    # Re-train the model to avoid overfit
    # re_checkpoint_path = LOGDIR + 're.cpp1.ckpt'
    # re_cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=re_checkpoint_path, save_weights_only=True, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    # re_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=20, mode='min', restore_best_weights=True)

    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # re_model = tf.keras.models.clone_model(model)
    # re_model.load_weights(latest)
    # re_model.layers[0].trainable = False
    
    # # # Implementation of multiple Metrics - ELIJA
    # METRICS = [
    #     tf.keras.metrics.categorical_accuracy,
    #     tf.keras.metrics.TruePositives(name='tp'),
    #     tf.keras.metrics.FalsePositives(name='fp'),
    #     tf.keras.metrics.TrueNegatives(name='tn'),
    #     tf.keras.metrics.FalseNegatives(name='fn'), 
    #     #tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    #     #tf.keras.metrics.Precision(name='precision'),
    #     #tf.keras.metrics.Recall(name='recall'),
    #     #tf.keras.metrics.AUC(name='auc'),
    # ]

    

    # if MULTIPLE_METRICS == True and CATEGORICAL_CROSSENTROPY == True:
    #     re_model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=METRICS) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    # if MULTIPLE_METRICS == True and CATEGORICAL_CROSSENTROPY == False:
    #     re_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    # if MULTIPLE_METRICS == False and CATEGORICAL_CROSSENTROPY == True:    
    #     re_model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['CategoricalAccuracy']) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
    # if MULTIPLE_METRICS == False and CATEGORICAL_CROSSENTROPY == False:
    #     re_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy']) # loss=tf.keras.losses.BinaryCrossentropy(), metrics=['CategoricalAccuracy'] - ELIJA
         
    # if IMBA_SOL == 0:
    #     re_history = re_model.fit(resampled_train_ds, shuffle=True, epochs=EPOCHS, steps_per_epoch=resampled_steps_per_epoch, verbose=2, validation_data=val_dataset, callbacks=[tensorboard_callback, re_cp_callback, re_early_stopping])
    # if IMBA_SOL == 1:
    #     re_history = re_model.fit(train_dataset, shuffle=True, epochs=EPOCHS, verbose=2, validation_data=val_dataset, class_weight=class_weight, callbacks=[tensorboard_callback, re_cp_callback, re_early_stopping])

    # Test model
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # model = initialize_model()
    # model.load_weights(latest)
    # loss,acc = model.evaluate(train_dataset, verbose=2)
    # loss,acc = model.evaluate(img_name_train, class_train, verbose=2)
    # training_logger.info("Restored model, accuracy: {:5.2f}%".format(100*acc))
    
    # eval_results = model.evaluate(val_dataset, batch_size=BATCH_SIZE, verbose=0)
    # training_logger.info('Evaluation:')
    # for name, value in zip(model.metrics_names, eval_results):
    #     training_logger.info(repr(name)+ ': '+ repr(value))

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    training_logger.info("Categorical training accuracy: "+repr(acc))
    training_logger.info("Categorical validation accuracy: "+repr(val_acc))
    training_logger.info("Training loss: "+repr(loss))
    training_logger.info("Validation loss: "+repr(val_loss))
    # training_logger.info(history)
    # training_logger.info(history.history)
    
    # Output lowest loss and best corresponding acc
    training_logger.info('Training dataset:')
    out_best_loss_and_acc(loss, acc)
    training_logger.info('Validation dataset:')
    out_best_loss_and_acc(val_loss, val_acc)

    # Re-training history
    # training_logger.info('Re-training History')
    # re_acc = re_history.history['categorical_accuracy']
    # re_val_acc = re_history.history['val_categorical_accuracy']

    # re_loss = re_history.history['loss']
    # re_val_loss = re_history.history['val_loss']

    # training_logger.info('Categorical re-training accuracy: ')
    # training_logger.info(re_acc)
    # training_logger.info('Categroical re-validation accuracy: ')
    # training_logger.info(re_val_acc)
    # training_logger.info('Re-Training loss: ')
    # training_logger.info(re_loss)
    # training_logger.info('Re-Validation loss: ')
    # training_logger.info(re_val_loss)

    # # Compare between first and re-training
    # if np.min(val_loss) < np.min(re_val_loss):
    #     training_logger.info('Load checkpoint from first training: '+repr(checkpoint_path))
    #     first_training_checkpoint = tf.train.load_checkpoint(checkpoint_path+'.index')
    #     model.load_weights(first_training_checkpoint)
    #     model.save_weights(filepath=checkpoint_dir)

    # Evaluate False classification cases of the model
    # training_logger.info('Training FN and FP')
    # out_fn_fp(model, train_dataset)
    training_logger.info('Val FN and FP')
    out_fn_fp(model, original_val_dataset.batch(BATCH_SIZE))

    # training_logger.info('After Re-training:')
    # training_logger.info('Re-training FN and FP')
    # out_fn_fp(re_model, train_dataset, class_train)
    # training_logger.info('Re-val FN and FP')
    # out_fn_fp(re_model, val_dataset, class_val)

    # Last layer weights
    # model_weights = get_weights(model)
    # model = load_model()
    # model layer's output
    # layer_output = model.layers[-2].output
    # layer_output = tf.keras.backend.get_value(layer_output)

    # get_layer_output = K.function(inputs = model.layers[0].input, outputs = model.layers[-2].output)
    # layer_output = get_layer_output(model.input)
    # training_logger.info(layer_output)

    #sys.stdout.close()