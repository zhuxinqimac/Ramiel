#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: main_lie.py
# --- Creation Date: 21-09-2020
# --- Last Modified: Mon 21 Sep 2020 16:23:30 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Main training for LieVAE.
Code borrowed from disentanglement_lib.
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
# Insert disentanglement_lib to path.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'disentanglement_lib'))
from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize import visualize_model
from disentanglement_lib.utils import resources
import argparse
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import gin.tf

from lie_models import LieVAE
from architectures import group_conv_encoder, group_deconv_decoder
from utils import _str_to_bool


def main():
    parser = argparse.ArgumentParser(description='Project description.')
    parser.add_argument('--result_dir',
                        help='Results directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/Ramiel/sweep')
    parser.add_argument('--study',
                        help='Name of the study.',
                        type=str,
                        default='unsupervised_study_v1')
    parser.add_argument('--model_gin',
                        help='Name of the gin config.',
                        type=str,
                        default='test_model.gin')
    parser.add_argument('--model_name',
                        help='Name of the model.',
                        type=str,
                        default='LieVAE')
    parser.add_argument('--hyps',
                        help='Hyperparameters of rec_spl_hes_lin_ncut_seed.',
                        type=str,
                        default='1_1_0_0_1_0')
    parser.add_argument('--overwrite',
                        help='Whether to overwrite output directory.',
                        type=_str_to_bool,
                        default=False)
    parser.add_argument('--dataset',
                        help='Dataset.',
                        type=str,
                        default='dsprites_full')
    parser.add_argument('--recons_type',
                        help='Reconstruction loss type.',
                        type=str,
                        default='bernoulli_loss')
    args = parser.parse_args()

    # 1. Settings
    study = reproduce.STUDIES[args.study]
    args.hyps = args.hyps.split('_')
    print()
    study.print_postprocess_config()
    print()
    study.print_eval_config()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Call training module to train the custom model.
    dir_name = "LieVAE-" + "-".join(args.hyps)
    output_directory = os.path.join(args.result_dir, dir_name)
    model_dir = os.path.join(output_directory, "model")
    gin_bindings = [
        "model.model = @" + args.model_name + "()",
        "LieVAE.hy_rec = " + args.hyps[0],
        "LieVAE.hy_spl = " + args.hyps[1],
        "LieVAE.hy_hes = " + args.hyps[2],
        "LieVAE.hy_lin = " + args.hyps[3],
        "LieVAE.hy_ncut = " + args.hyps[4],
        "model.random_seed = " + args.hyps[5],
        "dataset.name = '" + args.dataset + "'",
        "reconstruction_loss.loss_fn = @" + args.recons_type
    ]
    train.train_with_gin(model_dir, args.overwrite, [args.model_gin],
                         gin_bindings)

    # We fix the random seed for the postprocessing and evaluation steps (each
    # config gets a different but reproducible seed derived from a master seed of
    # 0). The model seed was set via the gin bindings and configs of the study.
    random_state = np.random.RandomState(0)

    # We extract the different representations and save them to disk.
    postprocess_config_files = sorted(study.get_postprocess_config_files())
    for config in postprocess_config_files:
        post_name = os.path.basename(config).replace(".gin", "")
        print("Extracting representation " + post_name + "...")
        post_dir = os.path.join(output_directory, "postprocessed", post_name)
        postprocess_bindings = [
            "postprocess.random_seed = {}".format(random_state.randint(2**32)),
            "postprocess.name = '{}'".format(post_name)
        ]
        postprocess.postprocess_with_gin(model_dir, post_dir, args.overwrite,
                                         [config], postprocess_bindings)

    # Iterate through the disentanglement metrics.
    eval_configs = sorted(study.get_eval_config_files())
    blacklist = ['downstream_task_logistic_regression.gin']
    # blacklist = [
    # 'downstream_task_logistic_regression.gin', 'beta_vae_sklearn.gin',
    # 'dci.gin', 'downstream_task_boosted_trees.gin', 'mig.gin',
    # 'modularity_explicitness.gin', 'sap_score.gin', 'unsupervised.gin'
    # ]
    for config in postprocess_config_files:
        post_name = os.path.basename(config).replace(".gin", "")
        post_dir = os.path.join(output_directory, "postprocessed", post_name)
        # Now, we compute all the specified scores.
        for gin_eval_config in eval_configs:
            if os.path.basename(gin_eval_config) not in blacklist:
                metric_name = os.path.basename(gin_eval_config).replace(
                    ".gin", "")
                print("Computing metric " + metric_name + " on " + post_name +
                      "...")
                metric_dir = os.path.join(output_directory, "metrics",
                                          post_name, metric_name)
                eval_bindings = [
                    "evaluation.random_seed = {}".format(
                        random_state.randint(2**32)),
                    "evaluation.name = '{}'".format(metric_name)
                ]
                evaluate.evaluate_with_gin(post_dir, metric_dir,
                                           args.overwrite, [gin_eval_config],
                                           eval_bindings)

    # We visualize reconstructions, samples and latent space traversals.
    visualize_dir = os.path.join(output_directory, "visualizations")
    visualize_model.visualize(model_dir, visualize_dir, args.overwrite)


if __name__ == "__main__":
    main()
