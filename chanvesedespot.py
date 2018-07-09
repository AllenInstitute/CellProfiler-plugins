# coding=utf-8

"""
MergeObjects
===========

TODO

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


class ChanVeseDespot(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "ChanVeseDespot"

    variable_revision_number = 1

    def create_settings(self):
        super(ChanVeseDespot, self).create_settings()


    def settings(self):
        __settings__ = super(ChanVeseDespot, self).settings()

        return __settings__ + [
        ]

    def visible_settings(self):
        __settings__ = super(ChanVeseDespot, self).visible_settings()

        return __settings__ + [
        ]

    def run(self, workspace):
        self.function = despot

        super(ChanVeseDespot, self).run(workspace)


def despot(image)
