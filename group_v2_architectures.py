#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: group_v2_architectures.py
# --- Creation Date: 29-09-2020
# --- Last Modified: Thu 08 Oct 2020 22:50:29 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
GroupV2VAE architectures.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'disentanglement_lib'))
import math
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import gin.tf
from utils import get_return_v, split_latents


@gin.configurable("group_v2_deconv_decoder",
                  whitelist=[
                      "hy_ncut", "group_feats_size", "lie_alg_init_scale",
                      "lie_alg_init_type", "n_act_points"
                  ])
def group_v2_deconv_decoder(latent_tensor,
                            output_shape,
                            hy_ncut=1,
                            group_feats_size=gin.REQUIRED,
                            lie_alg_init_scale=gin.REQUIRED,
                            lie_alg_init_type=gin.REQUIRED,
                            n_act_points=gin.REQUIRED,
                            is_training=True):
    """Convolutional decoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)
    Here we add an extra linear mapping for group features extraction.

    Args:
        latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
        output_shape: Shape of the data.
        group_feats_size: The dimension of group features.
        is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
        Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
          pixel intensities.
        group_feats: Group features.
    """
    # del is_training

    lie_alg_basis_ls = []
    latent_dim = latent_tensor.get_shape().as_list()[-1]
    latents_in_cut_ls = split_latents(latent_tensor, hy_ncut=hy_ncut)  # [x0, x1]

    mat_dim = int(math.sqrt(group_feats_size))
    for i in range(latent_dim):
        init = tf.initializers.random_normal(0, lie_alg_init_scale)
        lie_alg_tmp = tf.get_variable('lie_alg_' + str(i),
                                      shape=[1, mat_dim, mat_dim],
                                      initializer=init)
        if lie_alg_init_type == 'oth':
            lie_alg_tmp = tf.matrix_band_part(lie_alg_tmp, 0, -1)
            lie_alg_tmp = lie_alg_tmp - tf.transpose(lie_alg_tmp,
                                                     perm=[0, 2, 1])
        lie_alg_basis_ls.append(lie_alg_tmp)
    lie_alg_basis = tf.concat(lie_alg_basis_ls,
                              axis=0)[tf.newaxis,
                                      ...]  # [1, lat_dim, mat_dim, mat_dim]

    lie_alg = 0
    lie_group = tf.eye(mat_dim, dtype=lie_alg_basis_ls[0].dtype)[tf.newaxis, ...]
    for i, lie_alg_basis_i in enumerate(lie_alg_basis_ls):
        lie_alg_tmp = lie_alg_basis_i * latent_tensor[:, i][..., tf.newaxis, tf.newaxis]
        lie_alg = lie_alg + lie_alg_tmp
        lie_group_tmp = tf.linalg.expm(
            lie_alg_tmp)  # [b, mat_dim, mat_dim]
        lie_group = tf.matmul(lie_group_tmp, lie_group)
    # if not is_training:
        # lie_alg_mul = latent_tensor[
            # ..., tf.newaxis, tf.
            # newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        # lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
        # lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]
    # else:

        # lie_group = tf.eye(
            # mat_dim,
            # dtype=latents_in_cut_ls[0].dtype)[tf.newaxis, ...]
        # lie_alg = 0
        # for latents_in_cut_i in latents_in_cut_ls:
            # lie_alg_mul_tmp = latents_in_cut_i[
                # ..., tf.newaxis, tf.newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
            # lie_alg_tmp = tf.reduce_sum(
                # lie_alg_mul_tmp,
                # axis=1)  # [b, mat_dim, mat_dim]
            # lie_alg = lie_alg + lie_alg_tmp
            # lie_group_tmp = tf.linalg.expm(
                # lie_alg_tmp)  # [b, mat_dim, mat_dim]
            # lie_group = tf.matmul(lie_group,
                                  # lie_group_tmp)

    transed_act_points_tensor = tf.reshape(lie_group, [-1, mat_dim * mat_dim])

    # lie_alg_mul = latent_tensor[
    # ..., tf.newaxis, tf.
    # newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
    # lie_alg = tf.reduce_sum(lie_alg_mul, axis=1)  # [b, mat_dim, mat_dim]
    # lie_group = tf.linalg.expm(lie_alg)  # [b, mat_dim, mat_dim]

    # act_init = tf.initializers.random_normal(0, 0.01)
    # act_points = tf.get_variable('act_points',
    # shape=[1, mat_dim, n_act_points],
    # initializer=act_init)
    # transed_act_points = tf.matmul(lie_group, act_points)
    # transed_act_points_tensor = tf.reshape(transed_act_points,
    # [-1, mat_dim * n_act_points])

    d1 = tf.layers.dense(transed_act_points_tensor, 256, activation=tf.nn.relu)
    d2 = tf.layers.dense(d1, 1024, activation=tf.nn.relu)
    d2_reshaped = tf.reshape(d2, shape=[-1, 4, 4, 64])

    d3 = tf.layers.conv2d_transpose(
        inputs=d2_reshaped,
        filters=64,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )

    d4 = tf.layers.conv2d_transpose(
        inputs=d3,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )

    d5 = tf.layers.conv2d_transpose(
        inputs=d4,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
    )
    d6 = tf.layers.conv2d_transpose(
        inputs=d5,
        filters=output_shape[2],
        kernel_size=4,
        strides=2,
        padding="same",
    )
    return tf.reshape(d6, [-1] + output_shape), lie_group, lie_alg_basis
