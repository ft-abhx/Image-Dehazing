from skimage.metrics import structural_similarity
import cv2
# 3. Load the two input images
imageA = cv2.imread("../outputImages/result.jpg")
imageB = cv2.imread("../original/5920.jpg")

# 4. Convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 5. Compute the Structural Similarity Index (SSIM) between the two
#    images, ensuring that the difference image is returned
(score, diff) = structural_similarity(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# 6. You can print only the score if you want
print(f"SSIM value is : {round(score,4)}")