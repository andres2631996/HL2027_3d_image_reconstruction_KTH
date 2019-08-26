"""
This module contains all contents for Task 3 of lab L2.1.

**author**: danjorg@kth.se

**intended time**: 1.5hrs


## Description

In this task you will implement a function ("nlmeans") performing *non-local means* filtering for an input image.

Fill in the empty spaces marked with a **To Do**. Try out different parameter settings for `nlmeans` as well as
different noise levels. Which are the settings that give the best results?

Are you able to answer the following questions:

**a**. What is the purpose of the scale parameter of the gaussian weighting? How does it affect the performance of
       the filter? It is the standard deviation of the Gaussian distribution, like how much intensities are affected in the filtration (what we consider similar intensities and what we consider different)

**b**. How does the local patch size (`patch_width`) affect the performance of the filter? Larger sizes provide a slower execution, but a more refined average value, as we take into account more neighborhoods

**c**. The summation formula in the lecture shows a sum over a local neighbourhood (denoted \Omega). In the best case
       this neighbourhood would be the whole image domain. How is this neighbourhood defined in the code below? What
       could be a reason not to use the whole image domain? The neighbourhood is defined with the variable size_nghb. It is not used as the whole image since the execution would be too slow.


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
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.transform import resize
from scipy.stats.distributions import norm


def nlmeans(_im: np.ndarray, sigma_w: float, patch_width: int, size_nghb: int):
    """
    Perform `non local means` filtering on given input image `_im`. The new pixel value is composed
    of all pixel values in a local neighbourhood with radius `size_nghb`.

    Summation of intensities in the local neighbourhood of a pixel (i, j) is done by computing the mean intensity
    of an image patch of size `patch_width` for each pixel in the neighbourhood and weighting each
    summand with the gaussian of the distance to the mean of the image patch at (i, j).

    :param _im: input image; 2D
    :type _im: np.ndarray
    :param sigma_w: scale of gaussian weighting
    :type sigma_w: float
    :param patch_width: radius of image patches for mean computation; in pixels
    :type patch_width: int
    :param size_nghb: radius of neighbourhood for which to compute the weighted sum
    :type size_nghb: int
    :return: filtered image
    :rtype: np.ndarray
    """

    # initialise output image #
    _res = np.zeros_like(_im)

    # add padding around the image #
    padding = size_nghb + patch_width
    _im = np.pad(_im,padding,mode="reflect")  # TODO: use numpy's `pad` command; choose a proper padding mode!

    # loop over core of original image #
    for i in range(padding, _res.shape[0] + padding):

        print("Processing row {}".format(i-padding))

        for j in range(padding, _res.shape[1] + padding):

            # store local patch around (i, j) #
            _center_patch = _im[i-patch_width:i+patch_width+1,
                                j-patch_width:j+patch_width+1]

            # initialise local, weighted sum #
            _sum = 0
            _total_weight = 0

            # loop over neighbourhood #
            for x_loc in range(i-size_nghb, i+size_nghb+1):
                for y_loc in range(j-size_nghb, j+size_nghb+1):

                    # extract local patch at (x_loc, y_loc) #
                    _loc_patch = _im[x_loc-patch_width:x_loc+patch_width+1,
                                y_loc-patch_width:y_loc+patch_width+1]  # TODO: extract image patch

                    # compute norm of differences between patches #
                    _dist = np.linalg.norm(_center_patch-_loc_patch)  # TODO: use numpy's linalg.norm on the patch difference
                    
                    # compute the weight of the pixel at (x_loc, y_loc) #
                    weight = norm.pdf(_dist,scale=sigma_w)  # TODO: apply gaussian on the distances for weighting
                    
                    # update the local sum for (i, j) #
                    _sum += weight * _im[x_loc,y_loc]  # TODO: update weighted summation
                    
                    # accumulate weights #
                    _total_weight += np.sum(weight)  # TODO: update weights for normalisation
                    
            # assign new pixel value as the particular weighted sum #
            _res[i-padding, j-padding] = _sum/_total_weight  # TODO: assign new pixel value and normalise by weights.

    return _res


if __name__ == "__main__":

    # define input image #
    im = sitk.ReadImage(os.path.abspath('image_head.dcm'))
    im = sitk.GetArrayViewFromImage(im).squeeze().astype(np.float)

    # resize the image (a smaller image reduces computation time during testing) #
    im = resize(im, output_shape=(100, 100))

    # impose noise #
    im = im + norm.rvs(scale=10, size=im.shape)  # TODO: choose a proper noise level. try various values.

    # apply non local means filtering #
    res = nlmeans(im, sigma_w= 50, patch_width=2, size_nghb=5)  # TODO: call the implemented function.

    # PLOTTING #
    f = plt.figure()

    plot_opts = dict(cmap="gray")

    ax1 = plt.subplot(3, 1, 1)
    p1 = ax1.imshow(im, **plot_opts)
    ax1.set_title("Original image")
    f.colorbar(p1)

    ax2 = plt.subplot(3, 1, 2)
    p2 = ax2.imshow(res, **plot_opts)
    ax2.set_title("Non-local means")
    f.colorbar(p2)

    ax3 = plt.subplot(3, 1, 3)
    p3 = ax3.imshow(res-im, **plot_opts)
    ax3.set_title("Difference")
    f.colorbar(p3)

    plt.show()
