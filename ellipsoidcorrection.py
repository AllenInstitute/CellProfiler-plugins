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
import skimage.morphology
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class EllipsoidCorrection(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "MergeObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(EllipsoidCorrection, self).create_settings()

    def settings(self):
        __settings__ = super(EllipsoidCorrection, self).settings()

        return __settings__

    def visible_settings(self):
        __settings__ = super(EllipsoidCorrection, self).visible_settings()

        return __settings__

    def run(self, workspace):

        super(EllipsoidCorrection, self).run(workspace)
