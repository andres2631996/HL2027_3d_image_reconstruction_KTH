"""
This file contains all contents for Task 1 in Lab2.1.

**author**: danjorg@kth.se

**intended time**: 0.5 hrs


## Description

The task in this lab is to implement two functions for performing filtering of an image.

The functions `filter2d` and `filter2d_separable` take an image as well as the filter coefficients as inputs
and return the filtered image.

 **filter2d**. This function expects two input arguments: input_image, filter_kernel. It performs filtering
               based on the 2D convolution of the image with a two-dimensional filter kernel.

 **filter2d_separable**. This function expects three input arguments: input_image, filter_kernel along x, filter_kernel along y.
                         It performs filtering based on two subsequent 1D convolutions along the rows and the columns
                         of the input image with one-dimensional filter kernels.

At the bottom of the file you find a section for testing of the implemented function. Do both of your implementations
yield the same result? They yield a slightly similar result, since the difference image is not exactly zero
What do you call a 2D filter kernel that can be implemented as two one-dimensional operations? A separable filter
What is meant by "padding" and why is it needed? Padding consists in extending the rows and columns of the image with reflected values so to be able to filter in the borders of the image


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
from scipy.stats.distributions import norm
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from skimage.transform import resize



def filter2d(_im: np.ndarray, _k: np.ndarray):
    """
    Perform 2D filtering of `_im` with given kernel `_k` in spatial domain.

    :param _im: image; 2D
    :type _im: np.ndarray
    :param _k: kernel; 2D
    :type _k: np.ndarray
    :return: filtered image; 2D
    :rtype: np.ndarray
    """

    # define size of border regions of the image (half of kernel width) #
    _l = tuple((i // 2 for i in _k.shape))

    # define output image as array #
    _res = np.zeros_like(_im)

    # add padding to input image #
    padding = _l
    _im = np.pad(_im,padding,mode='reflect')  # TODO: use numpy's pad function to apply padding here

    # loop over first image dimension of output image (_res) #
    for i in range(padding[0], _res.shape[0] + padding[0]):

        # loop over second image dimension of output image (_res) #
        for j in range(padding[1], _res.shape[1] + padding[1]):

            # choose image patch centered at (i, j) #
            _patch = _im[i-padding[0]:i+padding[1]+1,
                         j-padding[1]:j+padding[1]+1]

            # elementwise multiplication of flipped (!) kernel with image patch centered at (i, j) #
            _prod = np.multiply(np.flip(_k,axis=0),_patch)  # TODO: multiply kernel and image patch

            # sum up elementwise multiplication along both axes #
            _sum = np.sum(_prod)  # TODO: sum up the product

            # assign resulting pixel value #
            _res[i-padding[0],j-padding[1]] = _sum  # TODO: assign the new pixel value

    # return resulting output image #
    return _res


def filter2d_separable(_im: np.ndarray, _k1: np.ndarray, _k2: np.ndarray):
    """
    Perform 2D filtering of `im` with given 1D kernels `k1` and `k2` in spatial domain.

    :param _im: image; 2D
    :type _im: ndarray
    :param _k1: kernel; 1D
    :type _k1: ndarray
    :param _k2: kernel; 1D
    :type _k2: ndarray
    :return: filtered image; 2D
    :rtype: ndarray
    """

    # define size of border regions of the image (half of kernel width) #
    _l = (len(_k1)//2, len(_k2)//2)

    # define output image as array #
    _res = np.zeros_like(_im)

    # add padding to input image #
    padding = _l
    _im = np.pad(_im,padding,mode='reflect')   # TODO: add padding as above
    _im_tmp = np.zeros_like(_im)  # TODO: What is the purpose of this variable?? Obtain the intermediate image resulting from the convolution of the first 1D kernel with the original image

    # loop over first image dimension of output image (_res) #
    for i in range(padding[0], _res.shape[0] + padding[0]):

        # loop over second image dimension of output image (_res) #
        for j in range(0, _im.shape[1]):

            # choose image patch along first dimension #
            _patch = _im[i-padding[0]:i+padding[1]+1,j]  # TODO: pick one-dimensional patch along 1st dimension

            # elementwise multiplication of flipped kernel with image patch centered at (i, j) #
            _prod = _k1[::-1] * _patch

            # sum up elementwise multiplication along both axes #
            _sum = np.sum(_prod)

            # assign resulting pixel value #
            _im_tmp[i, j] = _sum

    # loop over first image dimension of output image (_res) #
    for i in range(padding[0], _res.shape[0] + padding[0]):

        # loop over second image dimension of output image (_res) #
        for j in range(padding[1], _res.shape[1] + padding[1]):

            # choose image patch along second dimension #
            _patch = _im_tmp[i,j-padding[0]:j+padding[1]+1]  # TODO: pick one-dimensional patch along second dimension

            # elementwise multiplication of flipped (!) kernel with image patch centered at (i, j) #
            _prod = _k2[::-1] * _patch

            # sum up elementwise multiplication along both axes #
            _sum = np.sum(_prod)

            # assign resulting pixel value #
            _res[i-padding[0], j-padding[1]] = _sum

    # return resulting output image #
    return _res


if __name__ == "__main__":

    # define test image #
    im = sitk.ReadImage(os.path.abspath('image_head.dcm'))
    im = sitk.GetArrayViewFromImage(im).squeeze().astype(np.float)

    # resize the image (convenience for testing the code) #
    im = resize(im, output_shape=(100, 100))

    # impose noise #
    im = im + norm.rvs(scale=0.1, size=im.shape)  # TODO: add noise to the image. Vary the scale.

    # define kernel (e.g. gaussian kernel) #
    k1 = np.array([1/3, 1/3, 1/3])
    k = np.outer(k1, k1.transpose())

    # apply filter function from above #
    res = filter2d(im,k)  # TODO: apply the 2d filter function
    res_sep = filter2d_separable(im, k1, k1.transpose())  # TODO: apply the 1d separable filter function

    # apply convolution from scipy for comparison #
    res2 = convolve(im, k)

    # PLOTTING #
    f = plt.figure()

    # define global settings for all plots #
    plot_ops = dict(cmap="gray")

    # plot original input image #
    ax1 = plt.subplot(3, 2, 1)
    p1 = ax1.imshow(im, **plot_ops)
    ax1.set_title('Original image')
    f.colorbar(p1)

    # plot customly filtered image #
    ax2 = plt.subplot(3, 2, 2)
    p2 = ax2.imshow(res, **plot_ops)
    ax2.set_title('Filtered image ("filter2d")')
    f.colorbar(p2)

    # plot scipy filtered image #
    ax4 = plt.subplot(3, 2, 3)
    p4 = ax4.imshow(res2, **plot_ops)
    ax4.set_title('Filtered image (scipy)')
    f.colorbar(p4)

    # plot difference image #
    ax3 = plt.subplot(3, 2, 4)
    p3 = ax3.imshow(im-res, **plot_ops)
    ax3.set_title('Difference image ("filter2d" - original)')
    f.colorbar(p3)

    # plot separable algorithm outcome #
    ax5 = plt.subplot(3, 2, 5)
    p5 = ax5.imshow(res_sep, **plot_ops)
    ax5.set_title('Separable approach ("filter2d_separable")')
    f.colorbar(p5)

    # plot difference between 2d and separable approach #
    ax6 = plt.subplot(3, 2, 6)
    p6 = ax6.imshow(res-res_sep, **plot_ops)
    ax6.set_title('Difference separable ("filter2d" - "filter2d_separable")')
    f.colorbar(p6)

    # show the plots #
    plt.show()
