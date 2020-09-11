# Import Person_detection_tan
from Person_detection_tan import tf, pd, tqdm, os, np, argparse, IMG_HEIGHT, IMG_WIDTH, str2bool, preprocess_raise_img_vector, load_model, image_features_extract_model, image_flatten

import testdata as testdata

import logging

from pathlib import Path

# import cv2
from skimage.filters import threshold_otsu, threshold_multiotsu

from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
try:
    from PIL import Image
except ImportError:
    import Image

# Set Constants
FILENAME = 'RAISE_all.csv'
RAISE_DIR = '/original/RAISE/'

# Variables
global MODEL_DIR
# global CHECKPOINT_DIR
global FIRST_ONLY
global TOTAL_IMAGES
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = '/scratch/projekt1/demo/lr-0.000001_bs-4_ep-160_ti-8155_nm-True_is-1_us-Tan_kr-0.01/tmp/'
# CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))
# PLOT_DIR = '/scratch/projekt1/submitSkript/plots/'
PLOT_DIR = 'plots/'
plot_path = ''

# Set up logging
# local_logger = logging.getLogger('Localization')
# local_logger.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M', filename='output_loc.log', filemode='w')
# logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M', filename='/scratch/projekt1/Source/localization.log', filemode='w')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s: %(message)s')
# console.setFormatter(formatter)
# logging.getLogger().addHandler(console)
# local_logger = logging.getLogger()
local_logger = logging.getLogger('localization')
for hdlr in local_logger.handlers[:]:  # remove all old handlers
    local_logger.removeHandler(hdlr)

def preprocess_raise_test_path(total_images = 0):
    all_img_path_vector = []

    raise_db = pd.read_csv(RAISE_DIR + FILENAME)

    image_paths = raise_db.File
    total_img = image_paths.shape[0]

    if total_images > 0:
        total_img = total_images
    if FIRST_ONLY:
        total_img = 50
    
    for row in tqdm(range(total_img)):
        for root, dirs, files in os.walk(RAISE_DIR + 'RaiseTest'): 
            for file in files:
                if file.endswith(str(image_paths.loc[row]) + '.TIF'):
                    all_img_path_vector.append(root+'/'+str(file))

    return all_img_path_vector

def preprocess_raise_test_binary(total_images = 0):
    all_img_binary_vector = []

    raise_db = pd.read_csv(RAISE_DIR + FILENAME)

    keywords = raise_db.Keywords
    image_paths = raise_db.File

    image_paths = raise_db.File
    total_img = image_paths.shape[0]
    
    if total_images > 0:
        total_img = total_images
    if FIRST_ONLY:
        total_img = 50
    
    for row in tqdm(range(total_img)):
        for root, dirs, files in os.walk(RAISE_DIR + 'RaiseTest'): 
            for file in files:  
                if file.endswith(str(image_paths.loc[row]) + '.TIF'):
                    if('people' in str(keywords.loc[row]) ):
                        all_img_binary_vector.append([1, 0])
                    else:
                        all_img_binary_vector.append([0, 1])

    return all_img_binary_vector

# Load model
# def load_model(model_dir = MODEL_DIR, checkpoint_dir = CHECKPOINT_DIR):
#     # # Load existing model
#     # model = tf.keras.models.load_model(MODEL_PATH + model_name, compile=False)
#     model = tf.keras.models.load_model(model_dir)

#     latest = tf.train.latest_checkpoint(checkpoint_dir)
#     model.load_weights(latest)

#     # # Print summary
#     local_logger.info(model.summary())

#     return model

def get_weights(model):
    # local_logger.info(model.trainable_variables)
    # trainable_variables = tf.Variable(model.trainable_variables)
    # model_weights = trainable_variables.eval()
    
    # local_logger.info('Layer weight: ')
    # local_logger.info('Trainable variables: ' + repr(model.trainable_variables))
    # local_logger.info('[0]: ' + repr(model.trainable_variables[0]))
    # local_logger.info('[1]: ' + repr(model.trainable_variables[1]))
    model_weights = np.asarray(model.trainable_variables[0].numpy())
    # local_logger.info('Model weights: ' + repr(model_weights))
    # local_logger.info(model_weights.shape)

    model_weights_2 = model_weights[:,1]
    # local_logger.info('w2: ' + repr(model_weights_2))
    # local_logger.info(model_weights_2.shape)
    model_weights_1 = model_weights[:,0]
    # local_logger.info('w1: ' + repr(model_weights_1))
    # local_logger.info(model_weights_1.shape)

    if WEIGHT_MATRIX == 'w1':   
        # local_logger.info('weight w1')
        return model_weights_1
    else:
        # local_logger.info('weight w2')
        return model_weights_2

# Get flatten layer output
def get_flat_img(img):
    if (FIRST_ONLY):
        layer_last_model = tf.keras.Model(model.inputs, model.layers[-1].output)
        layer_last_out = layer_last_model(img, training=False)
        local_logger.info('Layer [-1]: ' + repr(layer_last_out))
        local_logger.info('Layer [-1] shape: ' + repr(layer_last_out.numpy().shape))

    flatten_layer_model = tf.keras.Model(model.inputs, model.layers[-2].output)
    flatten_layer_out = flatten_layer_model(img, training=False)
    # local_logger.info('Layer [-2]: ' + repr(flatten_layer_out))
    # local_logger.info('Layer [-2] shape: ' + repr(flatten_layer_out.numpy().shape))

    if (FIRST_ONLY):
        layer_first_out = image_features_extract_model(img, training=False)
        local_logger.info('Layer [-3]: ' + repr(layer_first_out))
        local_logger.info('Layer [-3] shape: ' + repr(layer_first_out.numpy().shape))
    
    flat_img = flatten_layer_out.numpy()[0]
    # features_extract = image_features_extract_model(img)
    # flat_img = image_flatten(features_extract).numpy()[0]
    # local_logger.info(flat_img)
    return flat_img

# # Calculate the weight matrix to locate people
def get_2d_sum_mat(weight_matrix, layer_matrix):
    # local_logger.info('Weight matrix shape: ' + repr(weight_matrix.shape))
    # local_logger.info('Layer matrix shape: ' + repr(layer_matrix.shape))
    
    flat_product_mat = [a*b for a, b in zip(weight_matrix , layer_matrix)]
    flat_product_mat = np.asarray(flat_product_mat)
    # cubic_product_mat = flat_product_mat.reshape((9, 15, 2048))
    cubic_product_mat = flat_product_mat.reshape(9, 15, 2048)

    sum_mat = np.sum(cubic_product_mat, axis=2)

    if (FIRST_ONLY):
        # For debugging only
        cubic_layer_matrix = layer_matrix.reshape(9, 15, 2048)
        local_logger.info('Cubic layer after reshape: ' + repr(cubic_layer_matrix))
        local_logger.info('Cubic layer shape: ' + repr(cubic_layer_matrix.shape))

        local_logger.info('Flat product shape: ' + repr(flat_product_mat.shape))
        local_logger.info('Cubic product: ' + repr(cubic_product_mat))
        local_logger.info('Cubic product shape: ' + repr(cubic_product_mat.shape))

        local_logger.info('Result matrix: ' + repr(sum_mat))
        local_logger.info('Result matrix shape: ' + repr(sum_mat.shape))

    return sum_mat

# Detect people in one image
def detect_people(img, img_path):
    # local_logger.info(img.shape)
    # First, predict if image has people
    prediction = model(img, training=False).numpy()
    # local_logger.info(prediction)

    # Second, if image has peple then do the localisization
    if prediction[0][0] > prediction[0][1]:
        flat_img = get_flat_img(img)
        sum_mat = get_2d_sum_mat(model_weights, flat_img)
        # replace all negative values with 0
        no_neg_sum_mat = sum_mat.copy()
        no_neg_sum_mat[no_neg_sum_mat < 0] = 0

        # Using Otsu threshold to locate people
        # sum_mat = sum_mat.astype('float32')
        # max = np.max(sum_mat)
        # min = np.min(sum_mat)
        # raw_th = max - 0.4*(max - min)
        # otsu_th, otsu_mat = cv2.threshold(sum_mat, raw_th, max, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        # local_logger.info(max, min, otsu_th)
        # draw_img_plot(img_path, otsu_mat, otsu_th, PLOT_DIR)

        # thresh = threshold_otsu(sum_mat)
        thresh_arr = threshold_multiotsu(no_neg_sum_mat, classes=3)
        thresh = thresh_arr[-1]
        # local_logger.info('Otsu threshold value: ' + repr(thresh))
        # Considering threshold_multiotsu for better localization?

        draw_img_plot(img_path, sum_mat, thresh, plot_path)
        return 1
    else:
        return 0

# Get 1st image containing people
def get_first_detection(path_vector, class_vector, all_img_name_vector):
    # img_pos = -1
    # for index in range(len(class_vector)):
    #     if class_vector[index] == [1,0]:
    #         img_pos = index
    #         break
    
    # img = all_img_name_vector[img_pos]
    # return detect_people(img, path_vector[img_pos])
    
    total_imgs = len(path_vector)
    positive = 0
    for i in tqdm(range(total_imgs)):
        img_path = path_vector[i]
        img = all_img_name_vector[i]
        positive = detect_people(img, img_path)
        if positive == 1:
            break
    return positive

# Detect people in all test dataset
def detect_people_all(path_vector, class_vector, all_img_name_vector):
    total_imgs = len(path_vector)
    total_positives = 0
    for i in tqdm(range(total_imgs)):
        img_path = path_vector[i]
        # local_logger.info(img_path)
        img = all_img_name_vector[i]
        pos = detect_people(img, img_path)
        total_positives += pos
    return total_positives

# # Draw image and plot
def draw_img_plot(img_path, result, thres = 0.4, plot_dir = PLOT_DIR):
    my_dpi=100.
    temp_img = Image.open(img_path)
    temp_img = temp_img.resize((600, 360))
    # temp_img = np.asarray(temp_img)
    # local_logger.info(temp_img.shape)

    img_filename = os.path.basename(img_path)
    # local_logger.info(img_filename)

    fig = plt.figure(figsize=(15, 9),dpi=my_dpi)
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    # fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    
    # Set the gridding interval: here we use the major tick interval
    myInterval=40.
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.set_xticks(np.arange(0, 15*myInterval, myInterval))
    ax.grid(which='major', axis='both', linestyle='-')

    # Add the image
    ax.imshow(temp_img)
    # fig.savefig(PLOT_DIR + '_img.jpg', dpi=my_dpi)
    
    for (i, j), z in np.ndenumerate(result):
        row = (i + 0.5)*myInterval
        col = (j + 0.5)*myInterval
        if z < thres:
            text_color = 'w'
        else:
            text_color = 'r'
            rect = patches.Rectangle((j*myInterval,i*myInterval),myInterval,myInterval,linewidth=2,edgecolor='r',facecolor=(0,1,0,0.3))
            ax.add_patch(rect)
        ax.text(col, row, '{:0.4f}'.format(z), ha='center', va='center', color=text_color)

    # plt.show()
    fig.savefig(plot_dir + img_filename + '_plot.jpg', dpi=my_dpi)
    plt.close("all")
    

if (__name__ == "__main__"):
    PARSER = argparse.ArgumentParser()
    # Adding arguments for parser
    PARSER.add_argument('--model_dir', type=str, default=MODEL_DIR, help='Model and checkpoint dir')
    # PARSER.add_argument('--checkpoint', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='Checkpoint dir')
    PARSER.add_argument('--weight_matrix', type=str, default='w1', help='Which weight matrix to use: w1 or w2')
    PARSER.add_argument('--first_only', type=str2bool, default=False, help='Get first result only')
    PARSER.add_argument('--total_images', type=int, default=150, help='Defining size of loaded dataset')
    PARSER.add_argument('--plot_dir', type=str, default=PLOT_DIR, help='Plot dir')

    args = PARSER.parse_args()
    for name, value in args._get_kwargs():
        variable_name = name.upper()
        exec(variable_name + " = value")
        # if name=='model_dir':
        #     MODEL_DIR=value
        #     continue
        # if name=='checkpoint':
        #     CHECKPOINT_DIR=value
        #     continue
        # if name=='first_only':
        #     FIRST_ONLY=value
        #     continue
        # if name=='total_images':
        #     TOTAL_IMAGES=value
        #     continue
        # if name=='plot_dir':
        #     PLOT_DIR=value
        #     continue
        
    # Create plots directory
    plot_parent_dir = MODEL_DIR.rstrip('//').replace('/tmp', '/')
    plot_path = plot_parent_dir + PLOT_DIR
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    # Logging
    local_logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if FIRST_ONLY:
        fh = logging.FileHandler(plot_parent_dir + 'localization_first.log')
    else:
        fh = logging.FileHandler(plot_parent_dir + 'localization.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    local_logger.addHandler(ch)
    local_logger.addHandler(fh)
    
    # local_logger.info(FIRST_ONLY)
    # local_logger.info(TOTAL_IMAGES)
    # local_logger.info(MODEL_DIR)
    # Get model weights
    model = load_model(model_dir = MODEL_DIR, checkpoint_dir = MODEL_DIR)
    model_weights = get_weights(model)

    # Get test dataset vector
    local_logger.info('Processing dataset')
    total_images = TOTAL_IMAGES
    start_index = 300 #For demo purpose
    # path_vector = preprocess_raise_test_path(total_images)
    # class_vector = preprocess_raise_test_binary(total_images)
    # all_img_name_vector = preprocess_raise_img_vector(path_vector)
    path_vector = testdata.path_vector[start_index:total_images+start_index]
    class_vector = testdata.class_vector[start_index:total_images+start_index]
    image_vector = []
    for i in tqdm(range(len(path_vector))):
        img_path = path_vector[i]
        img = image.imread(img_path)
        img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = tf.expand_dims(img, axis=0)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        # local_logger.info(img.shape)
        image_vector.append(img)
    
    # image_vector = np.asarray(image_vector)
    # local_logger.info(len(path_vector))
    # local_logger.info(len(class_vector))
    # local_logger.info(image_vector.shape)

    if FIRST_ONLY:
        first = get_first_detection(path_vector, class_vector, image_vector)
    else:
        total_pos = detect_people_all(path_vector, class_vector, image_vector)
        local_logger.info('Localization is complete')
        local_logger.info('Number of true positives: '+repr(class_vector.count([1,0])))
        local_logger.info('Number of predicted positives: '+repr(total_pos))