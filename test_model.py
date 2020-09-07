#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: test_model.py
# --- Creation Date: 06-09-2020
# --- Last Modified: Sun 06 Sep 2020 21:31:21 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'disentanglement_lib'))
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize import visualize_model
import glob
import tensorflow.compat.v1 as tf
import gin.tf

from models import GroupVAE
from architectures import group_conv_encoder, group_deconv_decoder

# 1. Settings
with open("local_config.txt", "r") as f:
    base_path = f.readlines()[0].strip()
# base_path = "/project/xqzhu_dis/repo_results/Ramiel/test"
overwrite = True

hys = []
# for hy_rec in ["0.1", "0.5", "0.", "1.", "2."]:
# for hy_spl in ["0.1", "0.5", "0.", "1.", "2."]:
for hy_rec in ["0."]:
    for hy_mat in ["0."]:
        for hy_oth in ["0."]:
            for hy_spl in ["0."]:
                for rand_seed in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                    hys.append([hy_rec, hy_mat, hy_oth, hy_spl, rand_seed])

for hy_rec, hy_mat, hy_oth, hy_spl, rand_seed in hys:
    # 2. Train a group VAE model.
    # We use the same training protocol that we defined in model.gin but we use gin
    # bindings to train our custom VAE instead of the ordinary VAE.
    gin_bindings = [
        "model.random_seed = " + rand_seed,
        "GroupVAE.hy_rec = " + hy_rec,
        "GroupVAE.hy_mat = " + hy_mat,
        "GroupVAE.hy_oth = " + hy_oth,
        "GroupVAE.hy_spl = " + hy_spl
    ]
    # Call training module to train the custom model.
    path_custom_vae = os.path.join(base_path,
            "GroupVAE-"+"-".join([hy_rec, hy_mat, hy_oth, hy_spl, rand_seed]))
    train.train_with_gin(os.path.join(path_custom_vae, "model"), overwrite,
                         ["test_model.gin"], gin_bindings)

    # 3. We visualize reconstructions, samples and latent space traversals.
    for path in [path_custom_vae]:
        visualize_path = os.path.join(path, "visualizations")
        model_path = os.path.join(path, "model")
        visualize_model.visualize(model_path, visualize_path, overwrite)

    # 4. Extract the mean representation for these models.
    for path in [path_custom_vae]:
        representation_path = os.path.join(path, "representation")
        model_path = os.path.join(path, "model")
        postprocess_gin = ["postprocess.gin"]  # This contains the settings.
        # postprocess.postprocess_with_gin defines the standard extraction protocol.
        postprocess.postprocess_with_gin(model_path, representation_path,
                                         overwrite, postprocess_gin)

    # 5. Compute metrics for all models.
    metrics = glob.glob('../disentanglement_lib/disentanglement_lib/config/unsupervised_study_v1/metric_configs/*.gin')
    blacklist = ['downstream_task_logistic_regression.gin']
    for path in [path_custom_vae]:
        for metric in metrics:
            if os.path.basename(metric) not in blacklist:
                result_path = os.path.join(path, "metrics", os.path.basename(metric).replace('.gin', ''))
                representation_path = os.path.join(path, "representation")
                evaluate.evaluate_with_gin(representation_path,
                                           result_path,
                                           overwrite,
                                           gin_config_files=[metric])

# 6. Aggregate the results.
# ------------------------------------------------------------------------------
# In the previous steps, we saved the scores to several output directories. We
# can aggregate all the results using the following command.
pattern = os.path.join(base_path,
                       "*/metrics/*/results/aggregate/evaluation.json")
results_path = os.path.join(base_path, "results.json")
aggregate_results.aggregate_results_to_json(pattern, results_path)

# 7. Print out the final Pandas data frame with the results.
# ------------------------------------------------------------------------------
# The aggregated results contains for each computed metric all the configuration
# options and all the results captured in the steps along the pipeline. This
# should make it easy to analyze the experimental results in an interactive
# Python shell. At this point, note that the scores we computed in this example
# are not realistic as we only trained the models for a few steps and our custom
# metric always returns 1.
model_results = aggregate_results.load_aggregated_json_results(results_path)
print(model_results)
