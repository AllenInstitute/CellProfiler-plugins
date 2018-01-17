# coding=utf-8

"""
ApplyTwoPeakClippingPlanes
==========================

**ApplyTwoPeakClippingPlanes** merges objects below a certain threshold into its most prevalent, adjacent neighbor.

The output of this module is a object image of the same data type as the input.
**ApplyTwoPeakClippingPlanes** can be run *after* any labeling or segmentation module (e.g.,
**ConvertImageToObjects** or **Watershed**). Labels are preserved and, where possible, small
objects are merged into neighboring objects that constitute a majority of the small object's
border. This is useful for reversing over-segmentation and artifacts that result from seeding
modules.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
NO           YES          NO
============ ============ ===============

"""

import numpy as np
import scipy.signal
import logging

import cellprofiler.image
import cellprofiler.object
import cellprofiler.module
import cellprofiler.setting

log = logging.getLogger(__name__)


class ApplyTwoPeakClippingPlanes(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "ApplyTwoPeakClippingPlanes"

    variable_revision_number = 1

    def create_settings(self):
        super(ApplyTwoPeakClippingPlanes, self).create_settings()

        self.reference_name = cellprofiler.setting.ImageNameSubscriber(
            text="Reference Image",
            doc="Image to reference for dual intensity peaks"
        )

        self.top_padding = cellprofiler.setting.Integer(
            text="Top Padding",
            value=0,
            doc="Additional slices to keep beyond intensity peak"
        )

        self.bottom_padding = cellprofiler.setting.Integer(
            text="Bottom Padding",
            value=0,
            doc="Additional slices to keep beyond intensity peak"
        )

        self.accept_single = cellprofiler.setting.Binary(
            text="Accept single peak as bottom clipping",
            value=False,
            doc="If only a single peak is found, accept this as the bottom cut-off"
        )

    def settings(self):
        __settings__ = super(ApplyTwoPeakClippingPlanes, self).settings()

        return __settings__ + [
            self.reference_name,
            self.top_padding,
            self.bottom_padding,
            self.accept_single
        ]

    def visible_settings(self):
        __settings__ = super(ApplyTwoPeakClippingPlanes, self).visible_settings()

        return __settings__ + [
            self.reference_name,
            self.top_padding,
            self.bottom_padding,
            self.accept_single
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        object_set = workspace.object_set
        images = workspace.image_set

        x = object_set.get_objects(x_name)

        if not x.volumetric:
            raise NotImplementedError("This module is only compatible with 3D (volumetric) images, not 2D.")

        dimensions = x.dimensions
        y_data = x.segmented.copy()

        reference_name = self.reference_name.value
        reference = images.get_image(reference_name, must_be_grayscale=True)
        reference_data = reference.pixel_data

        z_median = np.median(reference_data, axis=[1, 2])
        # `argrelmax` always returns a tuple, but z_median is one dimensional
        local_maxima = scipy.signal.argrelmax(z_median)[0]
        num_maxima = len(local_maxima)

        if num_maxima == 1 and self.accept_single:
            # Single peak accepted as bottom clipping plane
            # Don't clip off anything from the top
            local_maxima = [local_maxima[0], -1]
        elif num_maxima != 2:
            log.warn("Unable to find only two maxima (found {}) - bypassing clipping operation".format(num_maxima))
            local_maxima = [0, -1]

        # Apply padding based on user preference
        # Ensure the clipping plane isn't beyond the array's index
        bottom_slice = max(local_maxima[0] - self.bottom_padding.value, 0)
        top_slice = min(local_maxima[1] - self.top_padding.value, len(z_median) - 1)

        # Apply to new object
        y_data[:bottom_slice, :, :] = 0
        y_data[top_slice:, :, :] = 0

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data
        objects.parent_image = x.parent_image

        workspace.object_set.add_objects(objects, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x.segmented

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
