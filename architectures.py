#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: architectures.py
# --- Creation Date: 06-09-2020
# --- Last Modified: Sun 06 Sep 2020 21:29:57 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from disentanglement_lib.methods.shared import architectures  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import losses  # pylint: disable=unused-import
from disentanglement_lib.methods.shared import optimizers  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import gin.tf


@gin.configurable("group_conv_encoder", whitelist=["group_feats_size"])
def group_conv_encoder(input_tensor,
                       num_latent,
                       group_feats_size=gin.REQUIRED,
                       is_training=True):
    """Convolutional encoder used in beta-VAE paper for the chairs data.

    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    Here we add an extra linear mapping for group features extraction.

    Args:
        input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
          build encoder on.
        num_latent: Number of latent variables to output.
        group_feats_size: The dimension of group features.
        is_training: Whether or not the graph is built for training (UNUSED).

    Returns:
        means: Output tensor of shape (batch_size, num_latent) with latent variable
          means.
        log_var: Output tensor of shape (batch_size, num_latent) with latent
          variable log variances.
        group_feats: Group features.
    """
    del is_training

    e1 = tf.layers.conv2d(
        inputs=input_tensor,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e1",
    )
    e2 = tf.layers.conv2d(
        inputs=e1,
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e2",
    )
    e3 = tf.layers.conv2d(
        inputs=e2,
        filters=64,
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e3",
    )
    e4 = tf.layers.conv2d(
        inputs=e3,
        filters=64,
        kernel_size=2,
        strides=2,
        activation=tf.nn.relu,
        padding="same",
        name="e4",
    )
    flat_e4 = tf.layers.flatten(e4)
    e5 = tf.layers.dense(flat_e4, 256, activation=tf.nn.relu, name="e5")

    # Group feats mapping.
    group_feats = tf.layers.dense(e5,
                                  group_feats_size,
                                  activation=None,
                                  name="group_feats_E")

    means = tf.layers.dense(group_feats,
                            num_latent,
                            activation=None,
                            name="means")
    log_var = tf.layers.dense(group_feats,
                              num_latent,
                              activation=None,
                              name="log_var")
    mat_dim = int(math.sqrt(group_feats_size))
    group_feats_mat = tf.reshape(group_feats, [-1, mat_dim, mat_dim])
    return means, log_var, group_feats_mat


@gin.configurable("group_deconv_decoder", whitelist=["group_feats_size"])
def group_deconv_decoder(latent_tensor,
                         output_shape,
                         group_feats_size=gin.REQUIRED,
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
    del is_training
    group_feats = tf.layers.dense(latent_tensor,
                                  group_feats_size,
                                  activation=None)

    d1 = tf.layers.dense(group_feats, 256, activation=tf.nn.relu)
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
    mat_dim = int(math.sqrt(group_feats_size))
    group_feats_mat = tf.reshape(group_feats, [-1, mat_dim, mat_dim])
    return tf.reshape(d6, [-1] + output_shape), group_feats_mat
