import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # shuts off the tensorflow warnings
import tensorflow as tf
from imutils import contours
from sympy import solve, Symbol, Eq, S

# constants
MODEL_PATH = "model.tflite"  # default model path
MATH_IMAGE_NAME = "test7.png"
ROI_IMAGE_NAME = "ROI.png"
IMG_WIDTH = 60
IMG_HEIGHT = 60
BATCH_SIZE = 32

# global variables
class_names = ['!', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'd', 'div', 'f', 'i', 'j', 'k', 'l', 'p', 'pi', 'q', 'sqrt', 'x', 'u', 'v', 'w', 'x', 'y', 'z']
operators = ['!', '(', ')', '+', '-', '=', 'div', 'sqrt']
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)  # loads model from path

def main():
    """
    This is the main function.
    It receives no parameters,
    and prints the result of an equation in a given image.
    """
    # changes to the image so it could be seperated to parts
    image = cv2.imread(MATH_IMAGE_NAME)
    string_to_solve = get_math_str_from_image(image)
    results = solve_string(string_to_solve)
    if len(results) > 0:  # if there are any solutions to the equation in the image,
        for result in results:  # prints all of the solutions
            print(result)
    else:
        print("No result(s) for this math problem.")
    return


def get_math_str_from_image(image):
    """
    This function gets an image
    and returns the equation in it as a string.
    """
    mask = np.zeros(image.shape, dtype=np.uint8)  # masking
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscaling
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # thresholding

    # finding and organizing in an array the contours of the objects in the image
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # finds contours of objects in image
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # checks which contours should be used
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")  # sorts the selected contours

    string_to_solve = ""
    for c in cnts:
        area = cv2.contourArea(c)
        if 0 < area:  # if contours create a valid image object
            string_to_solve += get_char_from_image_details(c, thresh, mask)  # extracts character from image object
    string_to_solve = correct_string(string_to_solve)  # corrects the string
    return string_to_solve


def get_char_from_image_details(cntrs, thrshld, msk):
    """
    This string gets the contours, threshold and mask of an image
    and returns the character inside the image.
    """
    x, y, w, h = cv2.boundingRect(cntrs)  # creates boundaries for the object in the image
    # adding a little bit more area to the bounding area for a more accurate prediction
    x -= 10
    y -= 10
    w += 20
    h += 20

    # takes a part from the image using the bounding area
    ROI = 255 - thrshld[y:y + h, x:x + w]
    cv2.drawContours(msk, [cntrs], -1, (255, 255, 255), -1)

    # saving the part as an individual image to reload it with utils for further preperations for model prediction
    cv2.imwrite(ROI_IMAGE_NAME, ROI)                
    img = tf.keras.utils.load_img(
        ROI_IMAGE_NAME, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )

    # saving the part as an individual image in the right size for checking properties
    ROI = cv2.imread(ROI_IMAGE_NAME)
    ROI = cv2.resize(ROI, [IMG_HEIGHT, IMG_WIDTH])
    cv2.imwrite(ROI_IMAGE_NAME, ROI)
        
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # creates a batch

    # model's prediction
    classify_lite = interpreter.get_signature_runner("serving_default")
    predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)

    # return of the character
    os.remove(ROI_IMAGE_NAME)
    char = str(class_names[np.argmax(score_lite)]) # adds the part to the string that will hold the whole math problem
    return char


def correct_string(string_to_solve):
    """
    This function gets a string
    and corrects it in order to make it solve-able.
    """
    new_string_to_solve = ""
    for char_index in range(len(string_to_solve)):
        if string_to_solve[char_index] == "-" and string_to_solve[char_index + 1] == "-":  # if there's an '=' sign, it should be replaced with ','.
                                                                                           # in our program, the '=' sign is presented as 2 '-' signs instead.
            new_string_to_solve += ","
        elif string_to_solve[char_index - 1] == "-" and string_to_solve[char_index] == "-":  # in case we already changed the '=' sign, we do nothing
            pass
        elif string_to_solve[char_index] == "x":  # 'x' can be either a variable, a symbol of 'times' or a symbol of 'by the power of' (2 'x' in a row).
                                                  # this system of ifs and elses checks the meanings of each 'x' combination in the string
            x_count = 0
            other_char_index = char_index
            # checks how many 'x's are there in a row
            while 0 < other_char_index < len(string_to_solve) and string_to_solve[other_char_index] == "x":
                x_count += 1
                other_char_index += 1
            if x_count == 1: # in case of one 'x' or multiplication of numbers
                if string_to_solve[other_char_index] in operators: # if one 'x' - an operator will follow it
                    new_string_to_solve += "*x"
                else: # else - multiplication of numbers
                    new_string_to_solve += "*"
            elif x_count == 2: # in case of multiplication of 'x' by a number or a number to the power of another number
                if char_index == 0 or string_to_solve[other_char_index].isdigit(): # if multiplication of 'x' by a number - a number will follow or 'x' will be in the beginning of the equation
                    new_string_to_solve += "x*"
                elif char_index == len(string_to_solve) or string_to_solve[char_index - 1].isdigit():
                    new_string_to_solve += "*x"
                else: # else - a number to the power of another number
                    new_string_to_solve += "**"
            elif x_count == 3: # if 'x' to the power of a number
                new_string_to_solve += "x**"
            elif x_count == 4: # if a number is multiplied by 'x', and 'x' is also to the power of another number
                new_string_to_solve += "*x**"
            char_index = other_char_index - 1
        else:
            new_string_to_solve += string_to_solve[char_index]  # adds character normally
    return new_string_to_solve 


def solve_string(string_to_solve):
    """
    This string gets a string that contains an equation to solve
    and returns the solution(s) to the equation in the string.
    """
    are_there_vars = False
    for char in string_to_solve:
        if char.isalpha():
            are_there_vars = True
            break
    if (are_there_vars):
        eq = S(string_to_solve)
        eq = Eq(eq[0], eq[1])
        symbols = []
        for char in string_to_solve:
            if char.isalpha() and not (Symbol(char) in symbols):
                symbols.append(Symbol(char))
        return solve(eq, symbols)
    else:
        return list(eval(string_to_solve))


if __name__ == "__main__":
    main()