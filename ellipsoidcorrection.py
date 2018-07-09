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

import ast
import numpy
import skimage
import skimage.io
import skimage.filters
import skimage.exposure

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class EllipsoidCorrection(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "EllipsoidCorrection"

    variable_revision_number = 1

    def create_settings(self):
        super(EllipsoidCorrection, self).create_settings()

        self.sigma_image = cellprofiler.setting.Text(
            text="Sigma for smoothing the image",
            value="(3, 21, 21)",
            doc="Specify this either as a single integer or a tuple of integers the same shape as the image"
        )

        self.sigma_correction = cellprofiler.setting.Text(
            text="Sigma for smoothing the correction ellipsoid",
            value="(20, 80, 80)",
            doc="Specify this either as a single integer or a tuple of integers the same shape as the image"
        )

        self.correction_factor = cellprofiler.setting.Float(
            text="Factor when applying the correction function",
            value=0.5,
            minval=0.,
            maxval=1.,
        )

    def settings(self):
        __settings__ = super(EllipsoidCorrection, self).settings()

        __settings__ += [
            self.sigma_image,
            self.sigma_correction,
            self.correction_factor
        ]

        return __settings__

    def visible_settings(self):
        __settings__ = super(EllipsoidCorrection, self).visible_settings()

        __settings__ += [
            self.sigma_image,
            self.sigma_correction,
            self.correction_factor
        ]

        return __settings__

    def run(self, workspace):
        self.function = ellipsoid_correct

        super(EllipsoidCorrection, self).run(workspace)


def ellipsoid_correct(image, sigma_image, sigma_correction, correction_factor):
    # Since we're asking for a tuple or an integer, we have to literal eval the string
    # This is safe against code injection
    sigma_image = ast.literal_eval(sigma_image)
    sigma_correction = ast.literal_eval(sigma_correction)

    smoothed_orig = skimage.filters.gaussian(image, sigma=sigma_image, mode='wrap')
    correction_ellipsoid = skimage.filters.gaussian(numpy.ones(image.shape), sigma=sigma_correction, mode='constant')

    smoothed_corrected = smoothed_orig * skimage.exposure.rescale_intensity(correction_ellipsoid)

    corrected_image = skimage.img_as_float(image) + (correction_factor * smoothed_corrected)

    # Need to rescale the intensity now to get it in valid float range
    corrected_image = skimage.exposure.rescale_intensity(corrected_image)

    return skimage.img_as_uint(corrected_image)
