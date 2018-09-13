# coding=utf-8

"""
RemoveHolesPlanewise
====================


|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
NO           YES          NO
============ ============ ===============
"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class RemoveHolesPlanewise(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "RemoveHolesPlanewise"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveHolesPlanewise, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size of holes to fill",
            value=1.0,
            doc="Holes smaller than this diameter will be filled. Note that for 3D\
            images this module operates volumetrically so diameters should be given in voxels"
        )

    def settings(self):
        __settings__ = super(RemoveHolesPlanewise, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveHolesPlanewise, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda image, diameter: fill_holes(image, diameter)

        super(RemoveHolesPlanewise, self).run(workspace)


def fill_holes(image, diameter):
    radius = diameter / 2.0

    image = image.astype(bool)

    factor = radius ** 2

    size = numpy.pi * factor

    return numpy.array([skimage.morphology.remove_small_holes(s, size) for s in image])
