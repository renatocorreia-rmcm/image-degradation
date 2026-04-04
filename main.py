import cv2
import numpy as np

img: np.ndarray = cv2.imread('assets/gam.jpg')

if img is not None:
    print(type(img))
    print(img.shape)
    print(img)

    cv2.imshow('gam junior', img)

    # 0 means wait indefinitely for a key press
    # Any other positive integer (e.g., 1000) waits for that number of milliseconds
    cv2.waitKey(0)

    # Closes all the OpenCV windows we created
    cv2.destroyAllWindows()
