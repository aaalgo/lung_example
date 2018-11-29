#!/usr/bin/env python3
import math
import sys
sys.path.append('../aardvark')
sys.path.append('build/lib.linux-x86_64-' + sys.version[:3])
import random
import numpy as np
import tensorflow as tf
import aardvark
import rpn3d
from zoo import net3d
from lung import *
import cpp

flags = tf.app.flags
FLAGS = flags.FLAGS

MIN_SPACING = 0.75
MAX_SPACING = 0.85
SZ = 128

class Stream:
    # this class generates training examples
    def __init__ (self, path, is_training):
        self.ones = np.ones((1, SZ, SZ, SZ, 1), dtype=np.float32)
        samples = []
        with open(path, 'r') as f:
            for l in f:
                samples.append(l.strip())
                pass
            pass

        self.samples = samples
        self.sz = len(samples)
        self.is_training = is_training
        self.reset()
        pass

    def reset (self):
        samples = self.samples
        is_training = self.is_training

        def generator ():
            while True:
                if is_training:
                    random.shuffle(samples)
                    pass
                for path in samples:
                    volume = H5Volume(path)
                    n = volume.annotation.shape[0]
                    # sample one random annotation
                    # TODO: what if there are multiple nearby annotations?
                    if n == 0:  # no nodule
                        continue
                    n = random.randint(0, n-1)
                    spacing = random.uniform(MIN_SPACING, MAX_SPACING)
                    sub, nodule = extract_nodule(volume, volume.annotation[n], spacing, (SZ, SZ, SZ))
                    # TODO augmentation?
                    a, p = cpp.encode(sub.images.shape, nodule)
                    images = sub.images[np.newaxis, :, :, :, np.newaxis]
                    pw = a[np.newaxis, :, :, :, np.newaxis]
                    a = a[np.newaxis, :, :, :, np.newaxis]
                    p = p[np.newaxis, :, :, :, np.newaxis, :]
                    # we should also generate pure negative samples
                    yield None, images, a, self.ones, p, pw
                if not self.is_training:
                    break
        self.generator = generator()
        pass

    def size (self):
        return self.sz

    def next (self):
        return next(self.generator)

class Model (aardvark.Model, rpn3d.BasicRPN3D):
    def __init__ (self):
        aardvark.Model.__init__(self)
        rpn3d.BasicRPN3D.__init__(self)
        pass

    def rpn_backbone (self, volume, is_training, stride):
        assert(stride == 1)
        net, s = net3d.unet(volume, is_training)
        return net

    def rpn_logits (self, net, is_training, channels):
        return tf.layers.conv3d_transpose(net, channels, 3, strides=1, activation=None, padding='SAME')

    def rpn_params (self, net, is_training, channels):
        return tf.layers.conv3d_transpose(net, channels, 3, strides=1, activation=None, padding='SAME')

    def build_graph (self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.images = tf.placeholder(tf.float32, shape=(None, SZ, SZ, SZ, FLAGS.channels), name="volume")

        self.build_rpn(self.images, self.is_training, (SZ, SZ, SZ))
        pass

    def create_stream (self, path, is_training):
        return Stream(path, is_training)

    def feed_dict (self, record, is_training = True):
        _, images, a, aw, p, pw = record
        return {self.images: images,
                self.gt_anchors: a,
                self.gt_anchors_weight: aw, 
                self.gt_params: p,
                self.gt_params_weight: pw, 
                self.is_training: is_training}
    pass


def main (_):
    FLAGS.channels = 1
    FLAGS.classes = 1
    FLAGS.db = 'luna16.list'
    FLAGS.val_db = None
    FLAGS.epoch_steps = 100
    FLAGS.ckpt_epochs = 1
    FLAGS.val_epochs = 1000
    FLAGS.model = "model"
    FLAGS.rpn_stride = 1
    FLAGS.rpn_params = 4
    model = Model()
    aardvark.train(model)
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

