import numpy as np
import cv2

# def is_fuzzy(img, threshold=100):
#     """Return True if the image is fuzzy (blurry) based on Laplacian variance."""
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
#     return laplacian_var < threshold

def load_image(filepath):
    """ Load an image from a file path. """
    img = cv2.imread(filepath, 0)
    if img is None:
        raise ValueError(f"Image at {filepath} could not be loaded.")
    return img

def sharpen_image(img):
    """ Makes the text in the image appear more prominently """
    # if is_fuzzy(img):
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def threshold_image(img):
    """ Convert the image to binary using thresholding."""
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thresh = 255 - img_thresh
    return img_thresh

import numpy as np
import cv2

def align_text(img: np.ndarray) -> np.ndarray:
    """
    Aligns the text in an image to be horizontal, correcting for skew and orientation.
    """
    # Find all non-black pixels to determine the text's bounding box and angle
    # The coordinates must be in (x, y) format for minAreaRect
    coords = np.column_stack(np.where(img > 0))
    coords = coords[:, ::-1] # Swap (row, col) to (x, y)

    # Return the original image if there's not enough text to align
    if len(coords) < 10:
        return img

    # Get the minimum area bounding rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    width, height = rect[1]

    # Simplify the angle correction. If the bounding box is taller than it is
    # wide, we know the text is on its side, so we add 90 degrees.
    if width < height:
        angle += 90
        
    # Get the rotation matrix and apply the affine transformation
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Check if the text might be upside down
    # (if more "ink" is in the bottom half of the image)
    h_aligned, _ = img_aligned.shape[:2]
    upper_half_sum = np.sum(img_aligned[:h_aligned//2, :])
    lower_half_sum = np.sum(img_aligned[h_aligned//2:, :])
    
    if lower_half_sum > upper_half_sum:
        # A 180-degree rotation is a flip on both axes
        img_aligned = cv2.flip(img_aligned, -1) 

    return img_aligned

def split_image_to_rows(img):
    """ Splits the image into rows based on horizontal projections. 
        Returns row indices of where the text is present. (list of lists)
        
        eg. 
        img[rows[3][0]:rows[3][-1], :] will return the 4th row of text in the image.
    """
    a = np.sum(img==255, axis=1)
    rows = []
    seg = []

    for i in range(len(a)):
        if a[i] > 0:
            seg.append(i)
        
        if (a[i]==0) and (len(seg) >= 5):
            rows.append(seg)
            seg = []

        if len(seg) > 0:
            rows.append(seg)
    return rows

def grayscale_image(img):
    """ Convert the image to grayscale. """
    if len(img.shape) == 2:  # Already grayscale
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def remove_noise(img):
    """ Remove noise only if the image is noisy. """
    # Define a small 2x2 kernel to target small speckles
    kernel = np.ones((2,2), np.uint8)
    
    cleaned_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    return cleaned_img

def to_white_on_black(img):
    if np.mean(img) > 127:
        img = 255 - img
    return img

def preprocess_image(filepath):
    """ Preprocess the image for text recognition. """
    #Need to add resizing
    img = load_image(filepath)
    img = grayscale_image(img)
    img = sharpen_image(img)
    img = remove_noise(img)
    img = threshold_image(img)
    img = align_text(img)
    
    rows = split_image_to_rows(img)
        
    return img, rows



#Testing the preprocessing functions
