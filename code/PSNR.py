from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
original = cv2.imread("../original/4561.jpg")
compressed = cv2.imread("../outputImages/result2.jpg")
value = PSNR(original, compressed)
print(f"PSNR value is {round(value,4)} dB")
       