import numpy
import numpy.random
import numpy.testing
import pytest

import applytwopeakclippingplanes


instance = applytwopeakclippingplanes.ApplyTwoPeakClippingPlanes


@pytest.fixture(scope="function")
def volume_labels():
    labels = numpy.zeros((10, 20, 20), dtype=numpy.uint16)

    # Make some objects that span the whole array
    labels[:, 1:3, 1:3] = 1
    labels[:, 4:6, 4:6] = 2
    labels[:, 8:12, 8:12] = 3

    return labels


@pytest.fixture(scope="function")
def two_peak_reference():
    # Note: this is out of order for easier broadcasting later
    reference = numpy.ones((20, 20, 10), dtype=numpy.uint16)

    # Create some "peaks"
    reference *= numpy.asarray([1, 7, 9, 6, 5, 6, 8, 7, 4, 2], dtype=numpy.uint16)
    reference *= 1000

    # Make the reference noisy
    reference = reference.T
    reference = numpy.random.normal(reference, 100)

    return reference.astype(numpy.uint16)


@pytest.fixture(scope="function")
def single_peak_reference():
    # Note: this is out of order for easier broadcasting later
    reference = numpy.ones((20, 20, 10), dtype=numpy.uint16)

    # Create one "peaks"
    reference *= numpy.asarray([1, 3, 5, 9, 7, 6, 5, 4, 3, 2], dtype=numpy.uint16)
    reference *= 1000

    # Make the reference noisy
    reference = reference.T
    reference = numpy.random.normal(reference, 100)

    return reference.astype(numpy.uint16)


@pytest.fixture(scope="function")
def two_peak_gradient_reference():
    # Note: this is out of order for easier broadcasting later
    reference = numpy.ones((20, 20, 10), dtype=numpy.uint16)

    # Create some "peaks"
    # The peaks here are exaggerated so that the gradient method
    # picks them up appropriately
    reference *= numpy.asarray([1, 1, 5, 9, 5, 3, 5, 9, 3, 3], dtype=numpy.uint16)
    reference *= 1000

    # Make the reference noisy
    reference = reference.T

    return reference.astype(numpy.uint16)


@pytest.fixture(scope="function")
def aposteriori_reference():
    # Note: this is out of order for easier broadcasting later
    reference = numpy.ones((20, 20, 10), dtype=numpy.uint16)

    # Create some "peaks"
    # There are 3 potential peaks in this array
    reference *= numpy.asarray([1, 7, 9, 5, 6, 8, 3, 5, 8, 3], dtype=numpy.uint16)
    reference *= 1000

    # Make the reference noisy
    reference = reference.T
    reference = numpy.random.normal(reference, 100)

    return reference.astype(numpy.uint16)


# This is necessary because: https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['single_peak_reference', 'aposteriori_reference'])
def bypass_meta_fixture(request):
    return request.getfixturevalue(request.param)


# See above for necessity of meta fixture
@pytest.fixture(params=['two_peak_reference|2', 'single_peak_reference|3'])
def max_only_meta_fixture(request):
    param_name, expected_bottom = request.param.split('|')
    return request.getfixturevalue(param_name), int(expected_bottom)


def test_median_two_peak(volume_labels, two_peak_reference, module, object_set_empty, objects_empty,
                         image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # Everything outside of the peaks should be trimmed
    expected[:2] = 0
    expected[-3:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_sum_two_peak(volume_labels, two_peak_reference, module, object_set_empty, objects_empty,
                      image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_SUM

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # Everything outside of the peaks should be trimmed
    expected[:2] = 0
    expected[-3:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_padding_out(volume_labels, two_peak_reference, module, object_set_empty, objects_empty,
                     image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN
    module.top_padding.value = 1
    module.bottom_padding.value = 1

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # Everything outside of the peaks should be trimmed
    # But since we've added padding it should go further than normal
    expected[:1] = 0
    expected[-2:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_padding_in(volume_labels, two_peak_reference, module, object_set_empty, objects_empty,
                    image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN
    module.top_padding.value = -1
    module.bottom_padding.value = -2

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # Everything outside of the peaks should be trimmed
    # But since we've added padding it should go further than normal
    expected[:4] = 0
    expected[-4:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_moving_average_two_peak(volume_labels, two_peak_reference, module, object_set_empty, objects_empty,
                                 image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN
    module.use_moving_average.value = True

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # With a window size of 3 these cutoffs should be the same
    expected[:2] = 0
    expected[-3:] = 0

    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.xfail(strict=True)
def test_moving_average_large_window(volume_labels, two_peak_reference, module, object_set_empty, objects_empty,
                                     image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN
    module.use_moving_average.value = True
    module.moving_average_size.value = 5

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # These should NOT be the same
    # The window is so large only one peak should be found
    expected[:2] = 0
    expected[-3:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_gradient_two_peak(volume_labels, two_peak_gradient_reference, module, object_set_empty,
                           objects_empty, image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = two_peak_gradient_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_SUM
    module.use_gradient.value = True

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    #
    expected[:2] = 0
    expected[-3:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_bypass_preserves(volume_labels, bypass_meta_fixture, module, object_set_empty, objects_empty,
                          image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    bypass_meta_fixture = bypass_meta_fixture.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = bypass_meta_fixture

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    # Since the operation should be bypassed, the input should equal the output
    expected = labels.copy()

    numpy.testing.assert_array_equal(actual, expected)


def test_aposteriori_triple_peak(volume_labels, aposteriori_reference, module, object_set_empty, objects_empty,
                                 image_set_empty, image_empty, workspace_empty):
    labels = volume_labels.copy()
    reference = aposteriori_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_APOSTERIORI

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # Everything outside of the peaks should be trimmed
    expected[:2] = 0
    expected[-5:] = 0

    numpy.testing.assert_array_equal(actual, expected)


def test_max_only(volume_labels, max_only_meta_fixture, module, object_set_empty,
                  objects_empty, image_set_empty, image_empty, workspace_empty):
    peak_reference, expected_bottom = max_only_meta_fixture
    labels = volume_labels.copy()
    reference = peak_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_MAX_ONLY

    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_MEDIAN

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # Everything below the expected bottom should be removed
    expected[:expected_bottom] = 0

    numpy.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('top_exclude', [x for x in range(4)])
@pytest.mark.parametrize('bottom_exclude', [x for x in range(4)])
def test_excluding_slices(volume_labels, top_exclude, bottom_exclude,
                          objects_empty, image_empty, module,
                          workspace_empty, object_set_empty, image_set_empty):
    # Note: this is out of order for easier broadcasting later
    parabolic_reference = numpy.ones((20, 20, 9), dtype=numpy.uint16)

    # Create a parabolic intensity curve so there are only ever two peaks
    # and they're always at the "ends
    parabolic_reference *= numpy.asarray([9, 7, 5, 3, 1, 3, 5, 7, 9], dtype=numpy.uint16)
    parabolic_reference *= 1000

    parabolic_reference = parabolic_reference.T.astype(numpy.uint16)

    labels = volume_labels.copy()
    reference = parabolic_reference.copy()

    objects_empty.segmented = labels
    image_empty.pixel_data = reference

    module.x_name.value = "InputObjects"
    module.y_name.value = "OutputObjects"
    module.reference_name.value = "example"

    module.peak_method.value = applytwopeakclippingplanes.PEAK_NAIVE
    module.aggregation_method.value = applytwopeakclippingplanes.METHOD_SUM
    module.top_exclude.value = top_exclude
    module.bottom_exclude.value = bottom_exclude

    module.run(workspace_empty)

    actual = object_set_empty.get_objects("OutputObjects").segmented

    expected = labels.copy()
    # The peaks should be the same slice as the excluded slices
    expected[:bottom_exclude] = 0
    if top_exclude != 0:
        expected[-top_exclude:] = 0

    numpy.testing.assert_array_equal(actual, expected)