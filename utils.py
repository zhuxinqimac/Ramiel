#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: utils.py
# --- Creation Date: 06-09-2020
# --- Last Modified: Mon 05 Oct 2020 16:07:33 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""
import math
import numpy as np
import tensorflow.compat.v1 as tf
import argparse


def get_return_v(x, topk=1):
    if (not isinstance(x, tuple)) and (not isinstance(x, list)):
        return x if topk == 1 else tuple([x] + [None] * (topk - 1))
    if topk > len(x):
        return tuple(list(x) + [None] * (topk - len(x)))
    else:
        if topk == 1:
            return x[0]
        else:
            return tuple(x[:topk])


def split_latents(x, minibatch_size=1, hy_ncut=1):
    # x: [b, dim]
    # b = minibatch_size
    # b = x.get_shape().as_list()[0]
    if hy_ncut == 0:
        return [x]
    b = tf.shape(x)[0]
    dim = x.get_shape().as_list()[1]
    split_idx = tf.random.uniform(shape=[b, hy_ncut],
                                  maxval=dim + 1,
                                  dtype=tf.int32)
    split_idx = tf.sort(split_idx, axis=-1)
    idx_range = tf.tile(tf.range(dim)[tf.newaxis, :], [b, 1])
    masks = []
    mask_last = tf.zeros([b, dim], dtype=tf.float32)
    for i in range(hy_ncut):
        mask_tmp = tf.cast(idx_range < split_idx[:, i:i + 1], tf.float32)
        masks.append(mask_tmp - mask_last)
        masks_last = mask_tmp
    mask_tmp = tf.cast(idx_range < split_idx[:, -1:], tf.float32)
    masks.append(1. - mask_tmp)
    x_split_ls = [x * mask for mask in masks]
    # mask_1 = tf.cast(idx_range < split_idx[:, tf.newaxis], tf.float32)
    # mask_2 = 1. - mask_1
    # return x * mask_1, x * mask_2
    return x_split_ls


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
