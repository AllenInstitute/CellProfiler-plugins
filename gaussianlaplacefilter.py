# coding=utf-8

"""
LaplacianOfGaussian
===================

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import numpy.random
import scipy.ndimage

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class LaplacianOfGaussian(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "LaplacianOfGaussian"

    variable_revision_number = 1

    def create_settings(self):
        super(LaplacianOfGaussian, self).create_settings()

        self.gaussian_sigma = cellprofiler.setting.Float(
            text="Standard deviation for Gaussian kernel",
            value=1.,
            doc="Sigma defines how 'smooth' the Gaussian kernal makes the image. Higher sigma means a smoother image."
        )

        self.planewise = cellprofiler.setting.Binary(
            text="Perform filter planewise",
            value=True,
        )

    def settings(self):
        __settings__ = super(LaplacianOfGaussian, self).settings()

        return __settings__ + [
            self.gaussian_sigma,
            self.planewise
        ]

    def visible_settings(self):
        __settings__ = super(LaplacianOfGaussian, self).visible_settings()

        return __settings__ + [
            self.gaussian_sigma,
            self.planewise
        ]

    def run(self, workspace):
        self.function = laplacian_of_gaussian

        super(LaplacianOfGaussian, self).run(workspace)


def _laplacian_of_gaussian(array, sigma):
    return scipy.ndimage.gaussian_laplace(array, sigma)


def laplacian_of_gaussian(input_image, sigma, planewise):
    array = input_image.copy()
    if len(array.ndim) == 3 and planewise:
        array = numpy.array([_laplacian_of_gaussian(xy, sigma) for xy in array])
    else:
        array = _laplacian_of_gaussian(array, sigma)

    return array

