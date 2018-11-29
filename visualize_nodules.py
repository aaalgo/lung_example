#!/usr/bin/env python3
from lung import *
from glob import glob
from gallery import Gallery

gal = Gallery('nodule_samples', cols=3)

def visualize (image, y, x, r):
    image = np.copy(image, order='C')
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(image, (int(x), int(y)), int(r+5), (0, 255, 0))
    return image

for path in glob('scratch/luna16/*.h5')[:20]:
    volume = H5Volume(path)
    for i in range(volume.annotation.shape[0]):
        print(path, i)
        sub, anno = extract_nodule(volume, volume.annotation[i], 0.8, [128, 128, 128], random_crop=True)
        anno = np.round(anno).astype(np.int32)
        z, y, x, r = anno
        cv2.imwrite(gal.next(), visualize(sub.images[z, :, :], y, x, r))
        cv2.imwrite(gal.next(), visualize(sub.images[:, y, :], z, x, r))
        cv2.imwrite(gal.next(), visualize(sub.images[:, :, x], z, y, r))
        pass
gal.flush()

