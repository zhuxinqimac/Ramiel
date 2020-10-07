#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: group_v2_models.py
# --- Creation Date: 29-09-2020
# --- Last Modified: Thu 08 Oct 2020 01:58:06 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Model group vae v2.
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


@gin.configurable("GroupV2VAE")
class GroupV2VAE(BaseVAE):
    """GroupV2VAE.
    Test
    """
    def __init__(
            self,
            hy_gmat=gin.REQUIRED,
            hy_hes=gin.REQUIRED,
            hy_lin=gin.REQUIRED):
        self.hy_gmat = hy_gmat
        self.hy_hes = hy_hes
        self.hy_lin = hy_lin

    def model_fn(self, features, labels, mode, params):
        """TPUEstimator compatible model function."""
        del labels
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        data_shape = features.get_shape().as_list()[1:]
        batch_size = tf.shape(features)[0]
        z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
        z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)

        z_sampled_sum = z_sampled[:batch_size // 2] + \
            z_sampled[batch_size // 2:]
        z_sampled_all = tf.concat([z_sampled, z_sampled_sum], axis=0)
        # z_sampled_all = z_sampled
        reconstructions, group_feats_G, lie_alg_basis = self.decode_with_gfeats(
            z_sampled_all, data_shape, is_training)

        per_sample_loss = losses.make_reconstruction_loss(
            features, reconstructions[:batch_size])
        reconstruction_loss = tf.reduce_mean(per_sample_loss)
        kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled,
                                       group_feats_G, lie_alg_basis,
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

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled,
                    group_feats_G, lie_alg_basis, batch_size):
        del z_mean, z_logvar, z_sampled
        mat_dim = group_feats_G.get_shape().as_list()[1]
        gfeats_G = group_feats_G[:batch_size]
        gfeats_G_sum = group_feats_G[batch_size:batch_size + batch_size // 2]

        gfeats_G_mul = tf.matmul(gfeats_G[:batch_size // 2],
            gfeats_G[batch_size // 2:batch_size])

        lie_alg_basis_square = lie_alg_basis * lie_alg_basis
        # [1, lat_dim, mat_dim, mat_dim]
        _, lat_dim, mat_dim, _ = lie_alg_basis.get_shape().as_list()
        lie_alg_basis_col = tf.reshape(lie_alg_basis,
                                       [lat_dim, 1, mat_dim, mat_dim])
        lie_alg_basis_mul = tf.matmul(lie_alg_basis, lie_alg_basis_col)
        lie_alg_basis_mask = 1. - tf.eye(
            lat_dim, dtype=lie_alg_basis_mul.dtype)[:, :, tf.newaxis, tf.newaxis]
        lie_alg_basis_mul = lie_alg_basis_mul * lie_alg_basis_mask

        gmat_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(gfeats_G_mul - gfeats_G_sum), axis=[1, 2]))
        hessian_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(lie_alg_basis_mul), axis=[2, 3]))
        lin_loss = tf.reduce_mean(tf.reduce_sum(lie_alg_basis_square, axis=[2, 3]))
        loss = self.hy_gmat * gmat_loss + self.hy_hes * hessian_loss + self.hy_lin * lin_loss
        # loss = self.hy_hes * hessian_loss + self.hy_lin * lin_loss
        return kl_loss + loss
