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

        self.upper_percentile = cellprofiler.setting.Float(
            text="Upper percentile intensity value cutoff",
            value=99.,
            minval=0.,
            maxval=100.
        )

        self.set_percentile = cellprofiler.setting.Float(
            text="Percentile value to set pixels beyond upper bound",
            value=50.,
            minval=0.,
            maxval=100.
        )

        self.log_gain = cellprofiler.setting.Float(
            text="Log adjustment gain",
            value=1,
        )

    def settings(self):
        __settings__ = super(EllipsoidCorrection, self).settings()

        __settings__ += [
            self.sigma_z,
            self.sigma_y,
            self.sigma_x,
            self.upper_percentile,
            self.set_percentile,
            self.log_gain
        ]

        return __settings__

    def visible_settings(self):
        __settings__ = super(EllipsoidCorrection, self).visible_settings()

        __settings__ += [
            self.sigma_z,
            self.sigma_y,
            self.sigma_x,
            self.upper_percentile,
            self.set_percentile,
            self.log_gain
        ]

        return __settings__

    def run(self, workspace):
        self.function = ellipsoid_correct

        super(EllipsoidCorrection, self).run(workspace)


def ellipsoid_correct(image, sig_z, sig_y, sig_x, upper_percentile, set_percentile, log_gain):
    sigmas = (sig_z, sig_y, sig_x)
    super_smoothed = (skimage.filters.gaussian(image, sigma=sigmas, mode='constant') /
                      skimage.filters.gaussian(numpy.ones(image.shape), sigma=sigmas, mode='constant'))
    skimage.util.io

    # This is doing the opposite of what I want
    corrected = skimage.img_as_float(image) * super_smoothed

    # Ensure all the values are positive
    trimmed = corrected + numpy.abs(corrected.min())
    upper = numpy.percentile(trimmed, upper_percentile)
    set_val = numpy.percentile(trimmed, set_percentile)
    trimmed[trimmed > upper] = set_val

    # Log scale the image
    # scaled = skimage.exposure.adjust_log(trimmed, gain=log_gain)

    # scaled = skimage.exposure.rescale_intensity(scaled)

    return skimage.img_as_uint(trimmed)
