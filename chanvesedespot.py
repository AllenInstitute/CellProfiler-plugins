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

import ast
import numpy
import weave
import scipy.ndimage
import skimage.filters
import skimage.morphology
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting

F_RANDOM = 'Random'
F_MEDIAN = 'Median'
F_MIN = 'Minimum'
EPS = numpy.finfo(numpy.float).eps


class ChanVeseDespot(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "ChanVeseDespot"

    variable_revision_number = 1

    def create_settings(self):
        super(ChanVeseDespot, self).create_settings()

        self.initial_sigma = cellprofiler.setting.Text(
            text="Initial smoothing sigma",
            value="(1, 2, 2)",
            doc="Specify this either as a single integer or a tuple of integers the same shape as the image"
        )

        self.iterations = cellprofiler.setting.Integer(
            text="Number of iterations for Chan-Vese segmentation",
            value=20,
            minval=1
        )

        self.alpha = cellprofiler.setting.Float(
            text="Chan-vese alpha",
            value=0.
        )

        self.threshold = cellprofiler.setting.Float(
            text="Chan-vese threshold",
            value=0.
        )

        self.maximum_dot_size = cellprofiler.setting.Integer(
            text="Maximum dot size",
            value=30000,
            doc="Anything larger than this size will not be removed/replaced"
        )

        self.fill_method = cellprofiler.setting.Choice(
            text="Method for replacing dots",
            choices=[F_MEDIAN, F_MIN, F_RANDOM],
            value=F_RANDOM
        )

    def settings(self):
        __settings__ = super(ChanVeseDespot, self).settings()

        return __settings__ + [
            self.initial_sigma,
            self.iterations,
            self.alpha,
            self.threshold,
            self.maximum_dot_size,
            self.fill_method
        ]

    def visible_settings(self):
        __settings__ = super(ChanVeseDespot, self).visible_settings()

        return __settings__ + [
            self.initial_sigma,
            self.iterations,
            self.alpha,
            self.threshold,
            self.maximum_dot_size,
            self.fill_method
        ]

    def run(self, workspace):
        self.function = despot

        super(ChanVeseDespot, self).run(workspace)


def despot(image, initial_sigma, iterations, alpha, threshold, maximum_dot_size, fill_method):
    # Smooth the original image
    sigma = ast.literal_eval(initial_sigma)
    smoothed = skimage.filters.gaussian(image, sigma=sigma, mode='wrap')

    # Create a per-slice initialization mask for the Chan-Vese segmentation
    cv_init_mask = numpy.copy(smoothed)
    for idx, z_slice in enumerate(smoothed):
        cv_mean = numpy.mean(z_slice)
        cv_init_mask[idx] = z_slice < cv_mean

    # Apply Chan-Vese segmentation
    (seg_image, phi_image, its_image) = chanvese3d(smoothed, cv_init_mask, max_its=iterations,
                                                   alpha=alpha, thresh=threshold)

    seg_dots = phi_image >= 0  # Boundaries are tru
    seg_background = phi_image < 0  # Background is true

    # Separate cell membrane and dots from the mask: the segmented image (seg_dots)
    # After the segmented image is obtained, it is time to separate cell membrane and dots
    # from the segmented images because Chan-Vese segmentation identify cell membrane and dots both as boundaries.
    # This is the procedure of separation of cell membrane and dots from the mask.
    #
    # 1. Obtain the area of boundaries regions.Regions with big area would likely be a cell membrane
    #    while regions with small area would likely be dots.
    #    In this sense, the region with the biggest area would be the cell membrane for sure.
    # 2. Set a cut of area which separate cell membrane and dots.
    #    If the area of a region is bigger than the cut, the region would be the cell membrane.
    #    The area cut is found by experimenting with variable area cut.
    # 3. With the cut, separate cell membrane and dots from the mask.
    #
    # For further info, please refer to http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
    # TODO: The original code has this done per slice - is that necessary?
    seg_dots_only = numpy.ones_like(seg_dots)
    for idx, z_seg in enumerate(seg_dots):
        seg_label_obj, seg_nb_label = scipy.ndimage.label(z_seg)
        seg_sizes = numpy.bincount(seg_label_obj.ravel())
        # Set the background to 0, we don't care about it
        seg_sizes[0] = 0
        # For testing
        print("Segmentation sizes for slice {}:".format(idx))
        print(seg_sizes)
        # Apply the filter for minimum sizes
        mask_sizes = seg_sizes < maximum_dot_size
        # Get the index of the larges area, which is most likely the membrane
        max_idx = numpy.argmax(seg_sizes)
        mask_sizes[max_idx] = False
        seg_dots_only[idx] = mask_sizes[seg_label_obj]

    # Fill the dots
    despot_image = numpy.copy(image)

    # it fills dots with the median of z-slice picture
    if fill_method == F_MEDIAN:
        image_median = numpy.median(image, axis=(1, 2))
        for z_slice, z_mask, z_median in zip(despot_image, seg_dots_only, image_median):
            z_slice[z_mask] = z_median

    # it fills dots with the minimum of 20th percentile truncated z-slice picture
    # not minimum of entire z-slice picture
    elif fill_method == F_MIN:
        image_min = numpy.percentile(image, 20, axis=(1, 2))
        for z_slice, z_mask, z_min in zip(despot_image, seg_dots_only, image_min):
            z_slice[z_mask] = z_min

    # it fills dots with the Gaussian random number generated
    # from mean and standard deviation of 2 dimensional z-slice of background image.
    else:
        for z_slice, z_mask, z_bkgd in zip(despot_image, seg_dots_only, seg_background):
            bkgd_mean = numpy.mean(z_slice[z_bkgd])  # the mean of background image
            bkgd_std = numpy.std(z_slice[z_bkgd])    # the standard deviation of background image
            # 2D Gaussian random array generated from the mean and the standard deviation above.
            z_slice[z_mask] = numpy.random.normal(bkgd_mean, bkgd_std, z_slice.shape)[z_mask]

    return despot_image


##################################################
# The code below was adapted from:
#    Shawn Lankton (www.shawnlankton.com)
##################################################


def chanvese3d(I, init_mask, max_its=20, alpha=0, thresh=0):
    """
    Region Based Active Contour Segmentation

    seg = region_seg(I,init_mask,max_its,alpha,display)

    :rtype: object
    :param I: 2D image
    :param init_mask: Initialization (1 = foreground, 0 = bg)
    :param max_its: Number of iterations to run segmentation for
    :param alpha: (optional) Weight of smoothing term higer = smoother.  default = 0.2
    :param display: (optional) displays intermediate outputs default = false
    :return: Final segmentation mask (1=fg, 0=bg)

    Description: This code implements the paper: "Active Contours Without
    Edges" By Chan Vese. This is a nice way to segment images whose
    foregrounds and backgrounds are statistically different and homogeneous.

    Example:
    img = imread('tire.tif');
    m = zeros(size(img));
    m(33:33+117,44:44+128) = 1;
    seg = region_seg(img,m,500);

    Coded by: Shawn Lankton (www.shawnlankton.com)
    """

    I = I.astype('float')

    # -- Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask)

    # --main loop
    its = 0
    stop = False
    prev_mask = init_mask
    c = 0

    while its < max_its and not stop:

        # get the curve's narrow band
        idx = numpy.flatnonzero(numpy.logical_and(phi <= 1.2, phi >= -1.2))

        if len(idx) > 0:
            # -- find interior and exterior mean
            upts = numpy.flatnonzero(phi <= 0)  # interior points
            vpts = numpy.flatnonzero(phi > 0)  # exterior points
            u = numpy.sum(I.flat[upts]) / (len(upts) + EPS)  # interior mean
            v = numpy.sum(I.flat[vpts]) / (len(vpts) + EPS)  # exterior mean

            F = (I.flat[idx] - u) ** 2 - (I.flat[idx] - v) ** 2  # force from image information
            curvature = get_curvature(phi, idx)  # force from curvature penalty

            dphidt = F / numpy.max(numpy.abs(F)) + alpha * curvature  # gradient descent to minimize energy

            # -- maintain the CFL condition
            dt = 0.45 / (numpy.max(numpy.abs(dphidt)) + EPS)

            # -- evolve the curve
            phi.flat[idx] += dt * dphidt

            # -- Keep SDF smooth
            phi = sussman(phi, 0.5)

            new_mask = phi <= 0
            c = convergence(prev_mask, new_mask, thresh, c)

            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

        else:
            break

    # -- make mask from SDF
    seg = phi <= 0  # -- Get mask from levelset

    return seg, phi, its


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# -- AUXILIARY FUNCTIONS ----------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def bwdist(a):
    """
    this is an intermediary function, 'a' has only True, False vals,
    so we convert them into 0, 1 values -- in reverse. True is 0,
    False is 1, distance_transform_edt wants it that way.
    """
    return scipy.ndimage.distance_transform_edt(a == 0)


# -- converts a mask to a SDF
def mask2phi(init_a):
    """

    :param init_a:
    :return:
    """
    phi = bwdist(init_a) - bwdist(1 - init_a) + init_a - 0.5
    return phi


# -- compute curvature along SDF
def get_curvature(phi, idx):
    """

    :param phi:
    :param idx:
    :return:
    """
    dimz, dimy, dimx = phi.shape
    zyx = numpy.array([numpy.unravel_index(i, phi.shape) for i in idx])  # get subscripts
    z = zyx[:, 0]
    y = zyx[:, 1]
    x = zyx[:, 2]

    # -- get subscripts of neighbors
    zm1 = z - 1;
    ym1 = y - 1;
    xm1 = x - 1;
    zp1 = z + 1;
    yp1 = y + 1;
    xp1 = x + 1;

    # -- bounds checking
    zm1[zm1 < 0] = 0;
    ym1[ym1 < 0] = 0;
    xm1[xm1 < 0] = 0;
    zp1[zp1 >= dimz] = dimz - 1;
    yp1[yp1 >= dimy] = dimy - 1;
    xp1[xp1 >= dimx] = dimx - 1;

    # -- get central derivatives of SDF at x,y
    dx = (phi[z, y, xm1] - phi[z, y, xp1]) / 2  # (l-r)/2
    dxx = phi[z, y, xm1] - 2 * phi[z, y, x] + phi[z, y, xp1]  # l-2c+r
    dx2 = dx * dx

    dy = (phi[z, ym1, x] - phi[z, yp1, x]) / 2  # (u-d)/2
    dyy = phi[z, ym1, x] - 2 * phi[z, y, x] + phi[z, yp1, x]  # u-2c+d
    dy2 = dy * dy

    dz = (phi[zm1, y, x] - phi[zp1, y, x]) / 2  # (b-f)/2
    dzz = phi[zm1, y, x] - 2 * phi[z, y, x] + phi[zp1, y, x]  # b-2c+f
    dz2 = dz * dz

    # (ul+dr-ur-dl)/4
    dxy = (phi[z, ym1, xm1] + phi[z, yp1, xp1] - phi[z, ym1, xp1] - phi[z, yp1, xm1]) / 4

    # (lf+rb-rf-lb)/4
    dxz = (phi[zp1, y, xm1] + phi[zm1, y, xp1] - phi[zp1, y, xp1] - phi[zm1, y, xm1]) / 4

    # (uf+db-df-ub)/4
    dyz = (phi[zp1, ym1, x] + phi[zm1, yp1, x] - phi[zp1, yp1, x] - phi[zm1, ym1, x]) / 4

    # -- compute curvature (Kappa)
    curvature = ((dxx * (dy2 + dz2) + dyy * (dx2 + dz2) + dzz * (dx2 + dy2) -
                  2 * dx * dy * dxy - 2 * dx * dz * dxz - 2 * dy * dz * dyz) /
                 (dx2 + dy2 + dz2 + EPS))

    return curvature


def mymax(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    return (a + b + numpy.abs(a - b)) / 2


# -- level set re-initialization by the sussman method
def sussman(D, dt):
    """

    :param D:
    :param dt:
    :return:
    """
    # forward/backward differences
    a = D - shiftR(D)  # backward
    b = shiftL(D) - D  # forward
    c = D - shiftD(D)  # backward
    d = shiftU(D) - D  # forward
    e = D - shiftF(D)  # backward
    f = shiftB(D) - D  # forward

    a_p = a
    a_n = a.copy()  # a+ and a-
    b_p = b
    b_n = b.copy()
    c_p = c
    c_n = c.copy()
    d_p = d
    d_n = d.copy()
    e_p = e
    e_n = e.copy()
    f_p = f
    f_n = f.copy()

    i_max = D.shape[0] * D.shape[1] * D.shape[2]
    code = """
           for (int i = 0; i < i_max; i++) {
               if ( a_p[i] < 0 ) { a_p[i] = 0; }
               if ( a_n[i] > 0 ) { a_n[i] = 0; }
               if ( b_p[i] < 0 ) { b_p[i] = 0; }
               if ( b_n[i] > 0 ) { b_n[i] = 0; }
               if ( c_p[i] < 0 ) { c_p[i] = 0; }
               if ( c_n[i] > 0 ) { c_n[i] = 0; }
               if ( d_p[i] < 0 ) { d_p[i] = 0; }
               if ( d_n[i] > 0 ) { d_n[i] = 0; }
               if ( e_p[i] < 0 ) { e_p[i] = 0; }
               if ( e_n[i] > 0 ) { e_n[i] = 0; }
               if ( f_p[i] < 0 ) { f_p[i] = 0; }
               if ( f_n[i] > 0 ) { f_n[i] = 0; }
            }
    """
    weave.inline(code,
                 ['i_max',
                  'a_p', 'a_n', 'b_p', 'b_n', 'c_p', 'c_n', 'd_p', 'd_n', 'e_p', 'e_n', 'f_p', 'f_n']
                 )

    dD = numpy.zeros(D.shape)
    D_neg_ind = numpy.flatnonzero(D < 0)
    D_pos_ind = numpy.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = numpy.sqrt(mymax(a_p.flat[D_pos_ind] ** 2, b_n.flat[D_pos_ind] ** 2)
                                    + mymax(c_p.flat[D_pos_ind] ** 2, d_n.flat[D_pos_ind] ** 2)
                                    + mymax(e_p.flat[D_pos_ind] ** 2, f_n.flat[D_pos_ind] ** 2)
                                    ) - 1

    dD.flat[D_neg_ind] = numpy.sqrt(mymax(a_n.flat[D_neg_ind] ** 2, b_p.flat[D_neg_ind] ** 2)
                                    + mymax(c_n.flat[D_neg_ind] ** 2, d_p.flat[D_neg_ind] ** 2)
                                    + mymax(e_n.flat[D_neg_ind] ** 2, f_p.flat[D_neg_ind] ** 2)
                                    ) - 1

    D = D - dt * numpy.sign(D) * dD

    return D


# -- whole matrix derivatives
def shiftD(M):
    """

    :param M:
    :return:
    """
    shift = M[:, range(1, M.shape[1]) + [M.shape[1] - 1], :]
    return shift


def shiftL(M):
    """

    :param M:
    :return:
    """
    shift = M[:, :, range(1, M.shape[2]) + [M.shape[2] - 1]]
    return shift


def shiftR(M):
    """

    :param M:
    :return:
    """
    shift = M[:, :, [0] + range(0, M.shape[2] - 1)]
    return shift


def shiftU(M):
    """

    :param M:
    :return:
    """
    shift = M[:, [0] + range(0, M.shape[1] - 1), :]
    return shift


def shiftF(M):
    """

    :param M:
    :return:
    """
    shift = M[[0] + range(0, M.shape[0] - 1), :, :]
    return shift


def shiftB(M):
    """

    :param M:
    :return:
    """
    shift = M[range(1, M.shape[0]) + [M.shape[0] - 1], :, :]
    return shift


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    """

    :param p_mask:
    :param n_mask:
    :param thresh:
    :param c:
    :return:
    """
    #     diff = p_mask - n_mask
    diff = numpy.logical_xor(p_mask, n_mask)
    n_diff = numpy.sum(numpy.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0

    return c
