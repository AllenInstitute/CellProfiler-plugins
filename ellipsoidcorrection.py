# coding=utf-8

"""
EllipsoidCorrection
===================

** TODO **

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
NO           YES          NO
============ ============ ===============

"""

import numpy
import skimage
import skimage.filters

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class EllipsoidCorrection(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "EllipsoidCorrection"

    variable_revision_number = 1

    def create_settings(self):
        super(EllipsoidCorrection, self).create_settings()

        self.sigma_z = cellprofiler.setting.Float(
            text="Sigma for z-axis",
            value=3.,
            minval=0
        )

        self.sigma_y = cellprofiler.setting.Float(
            text="Sigma for y-axis",
            value=21.,
            minval=0
        )

        self.sigma_x = cellprofiler.setting.Float(
            text="Sigma for x-axis",
            value=21.,
            minval=0
        )

    def settings(self):
        __settings__ = super(EllipsoidCorrection, self).settings()

        __settings__ += [self.sigma_z, self.sigma_y, self.sigma_x]

        return __settings__

    def visible_settings(self):
        __settings__ = super(EllipsoidCorrection, self).visible_settings()

        __settings__ += [self.sigma_z, self.sigma_y, self.sigma_x]

        return __settings__

    def run(self, workspace):

        super(EllipsoidCorrection, self).run(workspace)


def ellipsoid_correct(image, sig_z, sig_y, sig_x):
    sigmas = (sig_z, sig_y, sig_x)
    super_smoothed = (skimage.filters.gaussian(image, sigma=sigmas, mode='constant') /
                      skimage.filters.gaussian(numpy.ones(image.shape), sigma=sigmas, mode='constant'))

    corrected = skimage.img_as_float(image) - super_smoothed

    return skimage.img_as_uint(corrected)
