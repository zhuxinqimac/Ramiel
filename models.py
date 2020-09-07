#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: models.py
# --- Creation Date: 05-09-2020
# --- Last Modified: Mon 07 Sep 2020 18:12:24 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Model group vae.
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
from disentanglement_lib.methods.unsupervised.vae import BaseVAE, compute_gaussian_kl
from disentanglement_lib.methods.unsupervised.vae import make_metric_fn
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import gin.tf
from tensorflow.contrib import tpu as contrib_tpu
from utils import get_return_v, split_latents


@gin.configurable("GroupVAE")
class GroupVAE(BaseVAE):
    """GroupVAE.
    Test
    """
    def __init__(
            self,
            hy_rec=gin.REQUIRED,
            hy_mat=gin.REQUIRED,
            # hy_gmat=gin.REQUIRED,
            hy_oth=gin.REQUIRED,
            hy_spl=gin.REQUIRED):
        self.hy_rec = hy_rec
        self.hy_mat = hy_mat
        # self.hy_gmat = hy_gmat
        self.hy_oth = hy_oth
        self.hy_spl = hy_spl

    def model_fn(self, features, labels, mode, params):
        """TPUEstimator compatible model function."""
        del labels
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_shape = features.get_shape().as_list()[1:]
        batch_size = tf.shape(features)[0]
        z_mean, z_logvar, group_feats_E = self.gaussian_encoder_with_gfeats(
            features, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)

        z_sampled_sum = z_sampled[:batch_size // 2] + \
            z_sampled[batch_size // 2:]
        z_sampled_split_1, z_sampled_split_2 = split_latents(
            z_sampled, batch_size)
        z_sampled_all = tf.concat(
            [z_sampled, z_sampled_sum, z_sampled_split_1, z_sampled_split_2],
            axis=0)
        reconstructions, group_feats_G = self.decode_with_gfeats(
            z_sampled_all, data_shape, is_training)

        per_sample_loss = losses.make_reconstruction_loss(
            features, reconstructions[:batch_size])
        reconstruction_loss = tf.reduce_mean(per_sample_loss)
        kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled,
                                       group_feats_E, group_feats_G,
                                       batch_size)
        loss = tf.add(reconstruction_loss, regularizer, name="loss")
        elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = optimizers.make_vae_optimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())
            train_op = tf.group([train_op, update_ops])
            tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            tf.summary.scalar("elbo", -elbo)

            logging_hook = tf.train.LoggingTensorHook(
                {
                    "loss": loss,
                    "reconstruction_loss": reconstruction_loss,
                    "elbo": -elbo
                },
                every_n_iter=100)
            return contrib_tpu.TPUEstimatorSpec(mode=mode,
                                                loss=loss,
                                                train_op=train_op,
                                                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            return contrib_tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
                                             "regularizer", "kl_loss"), [
                                                 reconstruction_loss, -elbo,
                                                 regularizer, kl_loss
                                             ]))
        else:
            raise NotImplementedError("Eval mode not supported.")

    def gaussian_encoder_with_gfeats(self, input_tensor, is_training):
        """
        For training use.
        """
        return architectures.make_gaussian_encoder(input_tensor,
                                                   is_training=is_training)

    def gaussian_encoder(self, input_tensor, is_training):
        """Applies the Gaussian encoder to images without return features.

        Args:
          input_tensor: Tensor with the observations to be encoded.
          is_training: Boolean indicating whether in training mode.

        Returns:
          Tuple of tensors with the mean and log variance of the Gaussian encoder.
        """
        return get_return_v(
            architectures.make_gaussian_encoder(input_tensor,
                                                is_training=is_training), 2)

    def decode_with_gfeats(self, latent_tensor, observation_shape,
                           is_training):
        """Decodes the latent_tensor to an observation."""
        return architectures.make_decoder(latent_tensor,
                                          observation_shape,
                                          is_training=is_training)

    def decode(self, latent_tensor, observation_shape, is_training):
        """Decodes the latent_tensor to an observation without features."""
        return get_return_v(
            architectures.make_decoder(latent_tensor,
                                       observation_shape,
                                       is_training=is_training), 1)

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled, group_feats_E,
                    group_feats_G, batch_size):
        del z_mean, z_logvar, z_sampled
        mat_dim = group_feats_E.get_shape().as_list()[1]
        gfeats_G = group_feats_G[:batch_size]
        gfeats_G_sum = group_feats_G[batch_size:batch_size + batch_size // 2]
        gfeats_G_split_1 = group_feats_G[batch_size +
                                         batch_size // 2:2 * batch_size +
                                         batch_size // 2]
        gfeats_G_split_2 = group_feats_G[2 * batch_size + batch_size // 2:]

        gfeats_E_mul = tf.matmul(group_feats_E[:batch_size // 2],
                                 group_feats_E[batch_size // 2:])
        # gfeats_G_mul = tf.matmul(gfeats_G[:batch_size // 2],
        # gfeats_G[batch_size // 2:])
        gfeats_G_2 = tf.matmul(gfeats_G, gfeats_G, transpose_b=True)
        G_mat_eye = tf.eye(mat_dim,
                           dtype=gfeats_G_2.dtype,
                           batch_shape=[batch_size])
        gfeats_G_split_mul = tf.matmul(gfeats_G_split_1, gfeats_G_split_2)

        rec_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(group_feats_E - gfeats_G), axis=[1, 2]))
        mat_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(gfeats_E_mul - gfeats_G_sum), axis=[1, 2]))
        # gmat_loss = tf.reduce_mean(
        # tf.reduce_sum(tf.square(gfeats_G_mul - gfeats_G_sum), axis=[1, 2]))
        oth_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(gfeats_G_2 - G_mat_eye), axis=[1, 2]))
        spl_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(gfeats_G_split_mul - gfeats_G),
                          axis=[1, 2]))
        loss = self.hy_rec * rec_loss + self.hy_mat * mat_loss + \
            self.hy_oth * oth_loss + self.hy_spl * spl_loss
        return kl_loss + loss
