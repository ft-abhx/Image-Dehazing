import numpy as np
import cv2

def CalTransmission(HazeImg, Transmission, regularize_lambda, sigma):
    rows, cols = Transmission.shape

    KirschFilters = LoadFilterBank()

    # Normalize the filters
    for idx, currentFilter in enumerate(KirschFilters):
        KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

    # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
    WFun = []
    for idx, currentFilter in enumerate(KirschFilters):
        WFun.append(CalculateWeightingFunction(HazeImg, currentFilter, sigma))

    # Precompute the constants that are later needed in the optimization step
    tF = np.fft.fft2(Transmission)
    DS = 0

    for i in range(len(KirschFilters)):
        D = psf2otf(KirschFilters[i], (rows, cols))
        DS = DS + (abs(D) ** 2)

    # Cyclic loop  
    beta = 1                    
    beta_max = 2**8             
    beta_rate = 2*np.sqrt(2)    

    while(beta < beta_max):
        gamma = regularize_lambda / beta

        
        DU = 0
        for i in range(len(KirschFilters)):
            dt = circularConvFilt(Transmission, KirschFilters[i])
            u = np.maximum((abs(dt) - (WFun[i] / (len(KirschFilters)*beta))), 0) * np.sign(dt)
            DU = DU + np.fft.fft2(circularConvFilt(u, cv2.flip(KirschFilters[i], -1)))

       

        Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
        beta = beta * beta_rate
    return(Transmission)

def LoadFilterBank():
    KirschFilters = []
    KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, 5],   [-3, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, -3],  [5, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3],   [5, 0, -3],   [5, 5, -3]]))
    KirschFilters.append(np.array([[5, -3, -3],    [5, 0, -3],   [5, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, -3],     [5, 0, -3],   [-3, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, 5],      [-3, 0, -3],  [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, 5, 5],     [-3, 0, 5],   [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, -3, 5],    [-3, 0, 5],   [-3, -3, 5]]))
    KirschFilters.append(np.array([[-1, -1, -1],   [-1, 8, -1],  [-1, -1, -1]]))
    return(KirschFilters)

def CalculateWeightingFunction(HazeImg, Filter, sigma):

    # Computing the weight function

    HazeImageDouble = HazeImg.astype(float) / 255.0
    if(len(HazeImg.shape) == 3):
        Red = HazeImageDouble[:, :, 2]
        d_r = circularConvFilt(Red, Filter)

        Green = HazeImageDouble[:, :, 1]
        d_g = circularConvFilt(Green, Filter)

        Blue = HazeImageDouble[:, :, 0]
        d_b = circularConvFilt(Blue, Filter)

        WFun = np.exp(-((d_r**2) + (d_g**2) + (d_b**2)) / (2 * sigma * sigma))
    else:
        d = circularConvFilt(HazeImageDouble, Filter)
        WFun = np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
    return(WFun)

def circularConvFilt(Img, Filter):
    FilterHeight, FilterWidth = Filter.shape
    assert (FilterHeight == FilterWidth), 'Filter must be square in shape --> Height must be same as width'
    assert (FilterHeight % 2 == 1), 'Filter dimension must be a odd number.'

    filterHalsSize = int((FilterHeight - 1)/2)
    rows, cols = Img.shape
    PaddedImg = cv2.copyMakeBorder(Img, filterHalsSize, filterHalsSize, filterHalsSize, filterHalsSize, borderType=cv2.BORDER_WRAP)
    FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)
    Result = FilteredImg[filterHalsSize:rows+filterHalsSize, filterHalsSize:cols+filterHalsSize]

    return(Result)


def psf2otf(psf, shape):
   
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
 
    psf = zero_pad(psf, shape, position='corner')

   
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

   
    otf = np.fft.fft2(psf)

    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def zero_pad(image, shape, position='corner'):
    
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img