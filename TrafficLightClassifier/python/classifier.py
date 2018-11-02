import cv2 # computer vision library
import helpers # helper functions
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

IMAGE_DIR_TRAINING = "training/"
IMAGE_DIR_TEST = "test/"
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

    
def standardize_input(image):
    ##Resize image and pre-process so that all "standard" images are the same size  
    standard_im = cv2.resize(image, (32,32))
    return standard_im

def one_hot_encode(label):

    label_types = ['red', 'yellow', 'green']
    # Create a vector of 0's that is the length of the number of classes (3)
    one_hot_encoded = [0] * len(label_types)

    # Set the index of the class number to 1
    one_hot_encoded[label_types.index(label)] = 1 

    return one_hot_encoded

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

def redBrightness(image):

    #Crop image
    top = image[0:16, 0:32]

    # Convert to HSV
    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)

    #Average Saturation
    sat = int(np.sum(hsv[:,:,1]) / 512 * 2.0)

    #Lower Red Hue Range
    lowerBottom = np.array([150,sat,140])
    lowerTop = np.array([10, 255, 255])
    lowerMask = cv2.inRange(hsv, lowerBottom, lowerTop)

    #Upper Red Hue Range
    upperBottom = np.array([160, sat, 140])
    upperTop = np.array([179, 255, 255])
    upperMask = cv2.inRange(hsv, upperBottom, upperTop)

    #Combined Mask
    red_hue_image = cv2.addWeighted(lowerMask, 1.0, upperMask, 1.0, 0.0)
    blur = cv2.GaussianBlur(red_hue_image,(9,9), 2, 2)

    #Return brightess feauture
    return float(np.sum(blur) / 512)

def greenBrightness(image):

    #Crop image
    bottom = image[16:32, 0:32]

    # Convert to HSV
    hsv = cv2.cvtColor(bottom, cv2.COLOR_RGB2HSV)

    #Average Saturation
    sat = int(np.sum(hsv[:,:,1]) / 512 * 1.3)

    #Lower Red Hue Range
    lowerBottom = np.array([70,sat,140])
    lowerTop = np.array([100, 255, 255])
    lowerMask = cv2.inRange(hsv, lowerBottom, lowerTop)

    #Blur
    blur = cv2.GaussianBlur(lowerMask,(9,9), 2, 2)

    #Return brightess feauture
    return float(np.sum(blur) / 512)

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)
random.shuffle(STANDARDIZED_LIST)

def estimate_label(image):

    green = greenBrightness(image)
    red = redBrightness(image)
    yellow = redBrightness(image)
    #If Brightness Feature detects more red than green. It returns a red estimation.
    if red > green:
        return [1,0,0]
    #If Brightness Feature detects more green than red. It returns a green estimation.
    elif green > red:
        return [0,0,1]
    #Returns yellow estimation.
    else:
        return [0,1,0]

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_LIST)

# Accuracy calculations
total = len(STANDARDIZED_LIST)
num_correct = float(total) - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))
