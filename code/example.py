
import cv2

from Airlight import Airlight
from BoundCon import BoundCon
from CalTransmission import CalTransmission
from removeHaze import removeHaze

if __name__ == '__main__':
    HazeImg = cv2.imread('../Images/8180.jpg')



    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 
    C1 = 300        # Default value = 300 
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                 

    # Refine estimate of transmission
    regularize_lambda = 1  # Default value = 1  --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

    cv2.imshow('Original', HazeImg)
    cv2.imshow('Result', HazeCorrectedImg)
    cv2.waitKey(0)

    cv2.imwrite('../outputImages/result7.jpg', HazeCorrectedImg)
