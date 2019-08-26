"""
This file contains all contents for task 2 of lab L2.1.

**author**: danjorg@kth.se

**intended time**: 1.0hrs


## Description

In this task you should implement a function applying bilateral filtering to an input image.

The function `filter_bilateral` expects the image as well as the scale parameters for two gaussian kernels
for spatial and intensity based filtering, respectively. The output is the filtered image.

Fill in the empty spaces in the code below. Test you code by running this script and vary the parameters
of the filter function for different noise scenarios (i.e. different levels). Which settings give the best results? An intermediate value (around 10) for the spatial variance and a high value for the intensity variance (around 100)
Which parameter choice leads to a similar behaviour as ordinary gaussian smoothing? When the intensity variance is low enough


## Notes

 * Make sure to answer the questions given in instructions (above or in the code)! It is
   important that you understand the targeted contents of this lab.

 * As opposed to previous labs, this task is provided as an executable python script.
   The reason for that is that the implementations are quite straitforward to do
   and easier to execute directly using a python interpreter.
   In order to execute the file, open a command line, activate your respective
   python virtual environment setup for this course, and switch to the folder containing
   this script. Then type `python T1.py` and press <enter> to execute the code in this file.

   In case you still prefer to use a jupyter notebook for this task, just copy the
   code to a new notebook file and execute it as in the labs before.

"""


import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from skimage.transform import resize


def filter_bilateral(_im: np.ndarray, sigma_s: float, sigma_i: float):
    """
    Perform bilateral filtering on `im` with `k_s` in spatial dimensions and `k_i` in intensity dimension.

    :param _im: input image
    :type _im: ndarray
    :param sigma_s: sigma of gaussian spatial filter
    :type sigma_s: float
    :param sigma_i: sigma of gaussian in intensity direction
    :type sigma_i: float
    :return: filtered image
    :rtype: ndarray
    """

    # define size of border regions of the image (half of kernel width) #
    _l = int(4 * sigma_s)

    # define output image as array #
    _res = np.zeros_like(_im)

    # add padding to input image #
    padding = _l
    _im = np.pad(_im, padding, mode="reflect")

    # define gaussian kernel for spatial smoothing #
    distances = np.array([[np.sqrt(a**2 + b**2) for a in np.linspace(-_l, _l, 2*_l+1)]
                          for b in np.linspace(-_l, _l, 2*_l+1)])
    kernel = norm.pdf(distances, scale=sigma_s)

    # loop over first image dimension of output image (_res) #
    for i in range(padding, _res.shape[0] + padding):

        print("Processing row {}".format(i-padding))

        # loop over second image dimension of output image (_res) #
        for j in range(padding, _res.shape[1] + padding):

            # choose image patch centered at (i, j) #
            _patch = _im[i-padding:i+padding+1,j-padding:j+padding+1]  # TODO: pick a subregion of the input image

            # elementwise multiplication of flipped kernel with image patch centered at (i, j) #
            _weight = kernel * norm.pdf(_patch-_im[i,j],sigma_i)  # TODO: multiply spatial and intensity weights
            _prod = np.multiply(_weight,_patch)  # TODO: apply weights to image patch

            # sum up elementwise multiplication along both axes #
            _sum = np.sum(_prod, axis=(0, 1))  # TODO: obtain sum of all weighted values
            _total_weight = np.sum(_weight,axis=(0, 1))  # TODO: accumulate weights for normalisation

            # assign resulting pixel value #
            _res[i-padding,j-padding] = _sum/_total_weight  # TODO: update pixel value with the sum normalised by total weights

    # return resulting output image #
    return _res


if __name__ == "__main__":

    # define test image #
    im = sitk.ReadImage(os.path.abspath('image_head.dcm'))
    im = sitk.GetArrayViewFromImage(im).squeeze().astype(np.float)

    # resize the image (smaller image size is computationally advantageous during testing) #
    im = resize(im, output_shape=(100, 100))

    # impose noise #
    im = im + norm.rvs(scale=10, size=im.shape)  # TODO: vary noise level (i.e. the scale parameter)

    # apply filter function from above #
    res = filter_bilateral(im, sigma_s=10, sigma_i=100)  # TODO: apply the implemented function

    # PLOTTING #
    f = plt.figure()

    # plot original input image #
    ax1 = plt.subplot(2, 1, 1)
    p1 = ax1.imshow(im, vmin=im.min(), vmax=im.max(), cmap='gray')
    ax1.set_title('Original image')
    f.colorbar(p1)

    # plot customly filtered image #
    ax2 = plt.subplot(2, 1, 2)
    p2 = ax2.imshow(res, cmap='gray')
    ax2.set_title('Filtered image (custom)')
    f.colorbar(p2)

    plt.show()
