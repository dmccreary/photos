import cv2
import numpy as np

def change_background_to_white(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image '{image_path}'")
        return

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an all white background
    white_background = np.full(img.shape, 255, dtype=np.uint8)

    # Copy the person to the white background
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        white_background[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    # Save the output image
    cv2.imwrite(output_path, white_background)

# Example usage
change_background_to_white('path_to_input_image.jpg', 'path_to_output_image.jpg')

