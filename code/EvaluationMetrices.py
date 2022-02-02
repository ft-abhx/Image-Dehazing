import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity
import cv2
def PSNR(original, dehazed):
    mse = np.mean((original - dehazed) ** 2)
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, dehazed):
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(dehazed, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

original = cv2.imread("../Images/8180.jpg")
dehazed = cv2.imread("../outputImages/result7.jpg")
PSNR_value = PSNR(original, dehazed)
SSIM_score=SSIM(original, dehazed)
print(f"PSNR value is {round(PSNR_value,4)} dB")
print(f"SSIM score is {round(SSIM_score,4)}")