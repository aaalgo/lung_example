#!/usr/bin/env python3
import math
import os
from glob import glob
import copy
import cv2
from scipy.ndimage import zoom
import pydicom as dicom
import numpy as np
import h5py
import logging
import random

ROOT = os.path.abspath(os.path.dirname(__file__))

DICOM_STRICT = False

def dicom_error (dcm, msg, level=logging.ERROR):
    s = 'DICOM ERROR (%s): %s' % (dcm.filename, msg)
    if DICOM_STRICT and level >= logging.ERROR:
        raise Exception(s)
    else:
        logging.log(level, s)
    pass

class DicomSlice:
    # one dicom image
    # we use class to extract the following from DICOM
    #   - the image or pixel array
    #   - patient position of slice (x,y,z) and row, col directions
    #   - location of the slice which can be used to sort slices
    #   - Hounsfield scale
    #   - spacing
    def __init__ (self, path):
        dcm = dicom.read_file(dcm_path)
        # slope & intercept for rescaling to Hounsfield unit
        self.HU = [float(dcm.RescaleSlope), float(dcm.RescaleIntercept)]
        # filename as slice ID
        #self.sid = os.path.splitext(os.path.basename(dcm.filename))[0]
        #self.dcm = dcm
        self.image = dcm.pixel_array
        self.pixel_padding = None
        #tag = Tag(0x0020,0x0032)
        #print dcm[tag].value
        #print dcm.ImagePositionPatient
        #assert dcm[tag] == dcm.ImagePositionPatient

        # patient position of the slice is determined by
        # - origin (self.position)
        # - direction of the row
        # - direction of the col
        x, y, z = [float(v) for v in dcm.ImagePositionPatient]
        self.position = [x, y, z]
        rx, ry, rz, cx, cy, cz = [float(v) for v in dcm.ImageOrientationPatient]
        self.ori_row = [rx, ry, rz]
        self.ori_col = [cx, cy, cz]

        x, y = [float(v) for v in dcm.PixelSpacing]
        assert x == y
        self.spacing = x
        # Stage1: 4704 missing SliceLocation
        try:
            self.location = float(dcm.SliceLocation)
        except:
            dicom_error(dcm, 'Missing SliceLocation', level=logging.DEBUG)
            self.location = self.position[2]
            pass
        self.bits = dcm.BitsStored

        if False:
            # Non have SliceThickness
            tag = dicom.tag.Tag(0x0018, 0x0050)
            if not tag in dcm:
                dicom_error(dcm, 'Missing SliceThickness', level=logging.WARN)
            else:
                logging.info('Has SliceThickness: %s' % dcm.filename)
                self.thickness = float(dcm[tag].value)

        # ???, why the value is as big as 63536
        if False:
            # Stage1 data:
            # 4704 have padding, 126057 not, so skip this
            self.padding = None
            try:
                self.padding = dcm.PixelPaddingValue
            except:
                dicom_error(dcm, 'Missing PixelPaddingValue', level=logging.WARN)
                pass

        # sanity check
        #if dcm.PatientName != dcm.PatientID:
        #    dicom_error(dcm, 'PatientName is not dcm.PatientID')
        if dcm.Modality != 'CT':
            dicom_error(dcm, 'Bad Modality: ' + dcm.Modality)
        #if Tag(0x0008,0x0008) in dcm:
        #    if not 'AXIAL' in ' '.join(list(dcm.ImageType)).upper():
        #        dicom_error(dcm, 'Bad image type: ' + list(dcm.ImageType))

        ori_type_tag = dicom.tag.Tag(0x0010,0x2210)
        if ori_type_tag in dcm:
            ori_type = dcm[ori_type_tag].value
            if 'BIPED' != ori_type:
                dicom_error(dcm, 'Bad Anatomical Orientation Type: ' + ori_type)

        # location should roughly be position.z
        self.problematic_slice_location = abs(self.position[2] - self.location) > 10

        x, y, z = self.ori_row  # should be (1, 0, 0)
        if x < 0.9:
            dicom_error(dcm, 'Bad row orientation')
        x, y, z = self.ori_col  # should be (0, 1, 0)
        if y < 0.9:
            dicom_error(dcm, 'Bad col orientation')
        pass
    pass

'''
AXIAL, SAGITTAL, CORONAL = 0, 1, 2

VIEWS = [AXIAL, SAGITTAL, CORONAL]
VIEW_NAMES = ['axial', 'sagittal', 'coronal']

AXES_ORDERS = ([0, 1, 2],  # AXIAL
               [2, 1, 0],  # SAGITTAL
               [1, 0, 2])  # CORONAL
'''

class VolumeBase (object):
    # self.images       # ndarray
    # self.spacing      # float or tuple of 3
    # self.origin       # tuple of 3 !!! origin is never transposed!!!
    # self.annotation   # np.array of [[z, y, x, r]] in world coordinate
    # self.HU           # 

    def __init__ (self):
        # the field an implementation of Volume should support
        self.uid = None
        self.path = None
        self.images = None      # 3-D array
        self.spacing = None     #
        self.origin = None      # origin never changes
                                # under transposing
        self.annotation = np.array((0, 4), dtype=np.float32)  # [(z, y, x, r)]
        # We save the coefficients for normalize to
        # Hounsfield Units, and keep that updated
        # when normalizing
        self.HU = None          # (intercept, slope)

        self.orig_origin = None
        self.orig_spacing = None
        self.orig_shape = None
        pass

    def save_h5 (self, path):
        assert self.HU == [1.0, 0]
        assert len(self.origin) == 3
        assert len(self.spacing) == 3
        spacing = np.array(list(self.spacing), dtype=np.float32)
        assert len(self.origin) == 3
        origin = np.array(self.origin, dtype=np.float32)
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=self.images)
            f.create_dataset("spacing", data=spacing)
            f.create_dataset("origin", data=origin)
            f.create_dataset("annotation", data=self.annotation)
            pass
        pass

    def normalizeHU (self):
        assert not self.HU is None
        a, b = self.HU
        self.images *= a
        self.images += b
        self.HU = [1.0, 0]
        pass


    # rescale images so spacing along each axis is the give value
    def rescale (self, spacing = 0.8):
        scale = [v / spacing for v in self.spacing]
        self.images = zoom(self.images, scale)
        self.spacing = [spacing, spacing, spacing]
        pass
    pass

    # clip color values to [min_th, max_th] and then normalize to [min, max]
    def normalize (self, min=0, max=1, min_th = -1000, max_th = 400):
        assert self.images.dtype == np.float32
        if not min_th is None:
            self.images[self.images < min_th] = min_th
        if not max_th is None:
            self.images[self.images > max_th] = max_th
        m = min_th #np.min(self.images)
        M = max_th #np.max(self.images)
        scale = (1.0 * max - min)/(M - m)
        logging.debug('norm %f %f' % (m, M))
        self.images -= m
        self.images *= scale
        self.images += min
        # recalculate HU 
        #   I:  original image
        #   I': new image
        #   a'I' + b' = aI + b
        #   I' = (I-m) * scale + min
        #      = I*scale + (min - m * scale)
        #   so
        #   a'I*scale + (min - m * scale)*a' + b' = aI + b
        #   
        #   a' = a / scale
        #   b' = b + a'(m * scale -min)
        #      = b + a * (m - min/scale)
        if self.HU:
            a, b = self.HU
            #self.HU = (a * (M -m), b + a * m)
            self.HU = (a / scale, b + a * (m - min/scale))
        pass

    def normalize_8bit (self):
        self.normalizeHU()
        self.normalize(min_th=-1000,max_th=400,min=0,max=255)
        pass

    def normalize_16bit (self):
        self.normalizeHU()
        self.normalize(min_th=-1000,max_th=400,min=0,max=1400)
        pass
    pass


# The dicom files of a volume might have been merged from multiple image
# acquisitions and might not form a coherent volume
# try to merge multiple acquisitions.  If that's not possible, keep the one
# with most slices
def regroup_dicom_slices (dcms):

    def group_zrange (dcms):
        zs = [float(dcm.dcm.ImagePositionPatient[2]) for dcm in dcms]
        zs = sorted(zs)
        gap = 1000000
        if len(zs) > 1:
            gap = zs[1] - zs[0]
        return (zs[0], zs[-1], gap)

    acq_groups = {}
    for dcm in dcms:
        an = 0
        try:
            an = int(dcm.dcm.AcquisitionNumber)
        except:
            pass
        acq_groups.setdefault(an, []).append(dcm)	
        pass
    groups = acq_groups.values()
    if len(groups) == 1:
        return groups[0]
    # we have multiple acquisitions
    zrs = [group_zrange(group) for group in groups]
    zrs = sorted(zrs, key=lambda x: x[0])
    min_gap = min([zr[2] for zr in zrs])
    gap_th = 2.0 * min_gap
    prev = zrs[0]
    bad = False
    for zr in zrs[1:]:
        gap = zr[0] - prev[1]
        if gap < 0 or gap > gap_th:
            bad = True
            break
        if gap != min_gap:
            logging.error('bad gap')
        prev = zr
    if not bad:
        logging.error('multiple acquisitions merged')
        return dcms
    # return the maximal groups
    gs = max([len(group) for group in groups])
    acq_groups = {k:v for k, v in acq_groups.iteritems() if len(v) == gs}
    key = max(acq_groups.keys())
    group = acq_groups[key]
    print(acq_groups.keys(), key)
    logging.error('found conflicting groups. keeping max acq number, %d out of %d dcms' % (len(group), len(dcms)))
    return group

# All DiCOMs of a UID, organized
class DicomVolume (VolumeBase):
    def __init__ (self, path, uid=None, regroup = True):
        VolumeBase.__init__(self)
        self.uid = uid
        self.path = path
        #self.thumb_path = os.path.join(DATA_DIR, 'thumb', uid)
        # load path
        dcms = []
        for dcm_path in glob(os.path.join(self.path, '*.dcm')):
            try:
                boxed = Dicom(dcm_path)
                dcms.append(boxed)
            except:
                print(dcm.filename)
                raise
            assert dcms[0].spacing == boxed.spacing
            assert dcms[0].shape == boxed.shape
            assert dcms[0].ori_row == boxed.ori_row
            assert dcms[0].ori_col == boxed.ori_col
            if dcms[0].pixel_padding != boxed.pixel_padding:
                logging.warn('0 padding %s, but now %s, %s' %
                        (dcms[0].pixel_padding, boxed.pixel_padding, dcm.filename))
            #assert dcms[0].HU == boxed.HU
            #print boxed.HU
            pass
        assert len(dcms) >= 2

        if regroup: # this operation can be ignored when trying to understand the code
            dcms = regroup_dicom_slices(dcms)

        dcms.sort(key=lambda x: x.position[2])
        zs = []
        for i in range(1, len(dcms)):
            zs.append(dcms[i].position[2] - dcms[i-1].position[2])
            pass
        zs = np.array(zs)
        z_spacing = np.mean(zs)
        assert z_spacing > 0
        assert np.max(np.abs(zs - z_spacing)) * 100 < z_spacing

        #self.length = dcms[-1].position[2] - dcms[0].position[2]
        front = dcms[0]
        #self.sizes = (front.shape[0] * front.spacing, front.shape[1] * front.spacing, self.length)
        self.dcms = dcms

        images = np.zeros((len(dcms),)+front.image.shape, dtype=np.float32)
        HU = front.HU
        for i in range(len(dcms)):
            HU2 = dcms[i].HU
            images[i,:,:] = dcms[i].image
            if HU2 != HU:
                assert False
                logging.warn("HU: (%d) %s => %s, %s" % (i, HU2, HU, dcms[i].dcm.filename))
                images[i, :, :] *= HU2[0] / HU[0]
                images[i, :, :] += (HU2[1] - HU[1])/HU[0]
        #self.dcm_z_position = {}
        #for dcm in dcms:
        #    name = os.path.splitext(os.path.basename(dcm.dcm.filename))[0]
        #    self.dcm_z_position[name] = dcm.position[2] - front.position[2]
        #    pass
        # spacing   # z, y, x
        self.images = images
        self.spacing = (z_spacing, front.spacing, front.spacing)
        x, y, z = front.position
        self.origin = [z, y, z] #front.location
        self.view = AXIAL
        self.anno = None
        self.HU = HU
        self.orig_origin = copy.deepcopy(self.origin)
        self.orig_spacing = copy.deepcopy(self.spacing)
        # sanity check
        pass
    pass

# MHD volume
class MetaImageVolume (VolumeBase):
    def __init__ (self, path, uid=None):
        import SimpleITK as itk
        VolumeBase.__init__(self)
        self.uid = uid
        self.path = path
        if not os.path.exists(self.path):
            raise Exception('data not found for uid %s at %s' % (uid, self.path))
        pass
        #self.thumb_path = os.path.join(DATA_DIR, 'thumb', uid)
        # load path
        itkimage = itk.ReadImage(self.path)
        self.HU = (1.0, 0.0)
        self.images = itk.GetArrayFromImage(itkimage).astype(np.float32)
        #print type(self.images), self.images.dtype
        self.origin = list(reversed(itkimage.GetOrigin()))
        self.spacing = list(reversed(itkimage.GetSpacing()))
        self.view = AXIAL
        _, a, b = self.spacing
        #self.anno = LUNA_ANNO.get(uid, None)
        assert a == b
        # sanity check
        pass
    pass

class H5Volume (VolumeBase):
    def __init__ (self, path, uid=None):
        super().__init__()
        with h5py.File(path, 'r') as f:
            self.images = f["images"].value
            spacing = f["spacing"].value
            origin = f["origin"].value
            # sanity check
            if spacing.shape == (1,):
                self.spacing = spacing[0]
            elif spacing.shape == (3,):
                self.spacing = (spacing[0], spacing[1], spacing[2])
                pass
            assert origin.shape == (3,)
            self.origin = (origin[0], origin[1], origin[2])
            self.annotation = f['annotation'].value
        pass
    pass
pass

def extract_nodule (volume, nodule, spacing, size, random_crop=True):
    # nodule (z, y, x, r) in world coordinate
    # spacing is a float number
    # size (Z, Y, X) is target size

    # returns sub, nodule
    # -- sub is a sub-volume of the input volume that contains the given nodule
    #         and has given spacing and size
    # -- nodule = (z,y,x,r) is the location in sub-volume
    
    S = [target_size * spacing * 1.1 / from_spacing for target_size, from_spacing in zip(size, volume.spacing)]
    S = np.ceil(np.array(S)).astype(np.int32)
    # S is the size in original volume corresponding to target size + delta

    C = (np.array(nodule[:3]) - np.array(volume.origin)) / np.array(volume.spacing)
    # C is the center of nodule
    R = nodule[3] / np.array(volume.spacing)
    # R is the radius of nodule in 3 axes, and 3 numbers might be different because spacing is different

    LL = np.array(volume.images.shape)

    lb = np.clip(np.floor(C - R), 0, None)       # lower bound (inclusive) of nodule
    ub = np.minimum(np.ceil(C + R + 1), LL)      # upper bound (exclusive) of nodule
    ll = ub - lb
    # extend beyond the nodule


    padding = S - ll
    assert np.all(padding >= 0)
    assert np.all(S <= LL)  # size of sub-volume

    right_space = LL - ub
    #right_space >= right_padding = padding - left_padding
    # so left_padding >= padding - right_space
    left_min = np.clip(padding - right_space, 0, None)
    left_max = np.minimum(lb, padding)

    delta = np.round((left_max - left_min)/4)
    left_min += delta
    left_max -= delta
    assert np.all(left_min <= left_max)

    # left position after padding
    if random_crop:
        z, y, x = [int(lb[i]) - random.randint(int(left_min[i]), int(left_max[i])) for i in range(3)]
    else:
        left = np.round((left_min + left_max)/2)
        z, y, x = [int(lb[i]) - int(left[i]) for i in range(3)]

    assert z >= 0 and y >= 0 and x >= 0

    sub = VolumeBase()
    sub.images = volume.images[z:(z+S[0]), y:(y+S[1]), x:(x+S[2])]
    assert np.all(np.array(sub.images.shape) == S)
    sub.spacing = volume.spacing
    sub.origin = volume.origin + np.array([z, y, x]) * volume.spacing 
    sub.HU = volume.HU

    sub.rescale(spacing)

    size = np.array(size, dtype=np.int32)
    # the volume size is still slightly bigger than required size, trim the edge
    z, y, x = np.floor((np.array(sub.images.shape)- size) / 2).astype(np.int32)
    sub.images = sub.images[z:(z+size[0]), y:(y+size[1]), x:(x+size[2])]
    sub.origin = sub.origin + np.array([z, y, x]) * sub.spacing 
    sub.HU = volume.HU
    assert np.all(np.array(sub.images.shape) == size)

    C = (np.array(nodule[:3]) - np.array(sub.origin)) / spacing
    R = nodule[3] / spacing

    return sub, list(C) + [R]


