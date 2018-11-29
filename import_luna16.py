#!/usr/bin/env python3
import pickle
import subprocess as sp
from tqdm import tqdm
from lung import *

LUNA_DIR = os.path.join(ROOT, 'luna16')

def load_luna_dir_layout ():
    lookup = {}
    for i in range(10):
        sub = os.path.join(LUNA_DIR, 'subset%d' % i)
        for f in glob(os.path.join(sub, '*.mhd')):
            bn = os.path.splitext(os.path.basename(f))[0]
            #print bn, "=>", sub
            lookup[bn] = sub
            pass
        pass
    return lookup

def load_luna_annotations ():
    ALL = {}
    with open(os.path.join(LUNA_DIR, 'CSVFILES', 'annotations.csv'), 'r') as f:
        next(f)
        for l in f:
            uid, x, y, z, d = l.strip().split(',')
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(d)/2
            ALL.setdefault(uid, []).append([z, y, x, r])
            pass
        pass
    return ALL

def load_luna_meta ():
    cache_path = os.path.join('scratch/meta.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    logging.warn('loading luna meta data')
    meta = (load_luna_dir_layout(), load_luna_annotations())
    with open(cache_path, 'wb') as f:
        pickle.dump(meta, f)
    return meta

LUNA_DIR_LOOKUP, LUNA_ANNO = load_luna_meta()

class LunaVolume (VolumeBase):
    def __init__ (self, path, uid=None):
        import SimpleITK as itk
        VolumeBase.__init__(self)

        if path is None and not uid is None:
            path = os.path.join(LUNA_DIR_LOOKUP[uid], uid + '.mhd')
        self.uid = uid
        self.path = path
        if not os.path.exists(self.path):
            raise Exception('data not found for uid %s at %s' % (uid, self.path))
        pass
        #self.thumb_path = os.path.join(DATA_DIR, 'thumb', uid)
        # load path
        itkimage = itk.ReadImage(self.path)
        self.HU = [1.0, 0.0]
        self.images = itk.GetArrayFromImage(itkimage).astype(np.float32)
        #print type(self.images), self.images.dtype
        self.origin = list(reversed(itkimage.GetOrigin()))
        self.spacing = list(reversed(itkimage.GetSpacing()))
        _, a, b = self.spacing
        self.annotation = LUNA_ANNO.get(uid, [])
        assert a == b
        # sanity check
        pass
    pass

if __name__ == '__main__':
    sp.check_call('mkdir -p scratch/luna16', shell=True)
    for uid, path in tqdm(LUNA_DIR_LOOKUP.items()):
        volume = LunaVolume(None, uid=uid)
        volume.save_h5("scratch/luna16/%s.h5" % uid)
        pass

