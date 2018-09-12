# coding=utf-8

"""
ErodeObjects
==============

**ErodeObjects** shrinks bright shapes in an image. See `this tutorial`_ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

.. _this tutorial: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_morphology.html#erosion

"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
from cellprofiler.modules._help import HELP_FOR_STREL


class ErodeObjects(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "ErodeObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(ErodeObjects, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement(allow_planewise=True,
                                                                           doc=HELP_FOR_STREL)


        self.planewise = cellprofiler.setting.Binary(
            text="Planewise fill",
            value=False,
            doc="""\
Select "*{YES}*" to fill objects on a per-plane level. 
This will perform the hole filling on each plane of a 
volumetric image, rather than on the image as a whole. 
This may be helpful for removing seed artifacts that 
are the result of segmentation.
**Note**: Planewise operations will be considerably slower.
""".format(**{
                "YES": cellprofiler.setting.YES
            })
        )

    def settings(self):
        __settings__ = super(ErodeObjects, self).settings()

        return __settings__ + [
            self.structuring_element,
            self.planewise
        ]

    def visible_settings(self):
        __settings__ = super(ErodeObjects, self).settings()

        return __settings__ + [
            self.structuring_element,
            self.planewise
        ]

    def run(self, workspace):

        x = workspace.object_set.get_objects(self.x_name.value)

        is_strel_2d = self.structuring_element.value.ndim == 2

        is_img_2d = x.segmented.ndim == 2

        if is_strel_2d and not is_img_2d:

            self.function = erode_objects

        elif not is_strel_2d and is_img_2d:

            raise NotImplementedError("A 3D structuring element cannot be applied to a 2D image.")

        else:

            self.function = erode_objects

        super(ErodeObjects, self).run(workspace)


def _erode(labels, strel):
    array = numpy.zeros_like(labels)
    # Iterate through each label and dilate it
    for n in numpy.unique(labels):
        if n == 0:
            continue

        eroded_mask = skimage.morphology.binary_erosion(labels == n, selem=strel)
        array[eroded_mask] = n
    return array


def erode_objects(labels, strel, planewise):
    # Only operate planewise if image is 3D and planewise requested
    if planewise and labels.ndim != 2 and labels.shape[-1] not in (3, 4):
        return numpy.array([_erode(x, strel) for x in labels])
    return _erode(labels, strel)
