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
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow.compat.v1 as tf
import gin.tf

from models import GroupVAE
from architectures import group_conv_encoder, group_deconv_decoder

# 1. Settings
base_path = "/mnt/hdd/repo_results/Ramiel/test"
overwrite = True

# 2. Train a group VAE model.
# We use the same training protocol that we defined in model.gin but we use gin
# bindings to train our custom VAE instead of the ordinary VAE.
gin_bindings = [
    "model.random_seed = 0", "GroupVAE.hy_rec = 1.", "GroupVAE.hy_mat = 1.",
    "GroupVAE.hy_oth = 1.", "GroupVAE.hy_spl = 1."
]
# Call training module to train the custom model.
path_custom_vae = os.path.join(base_path, "GroupVAE")
train.train_with_gin(os.path.join(path_custom_vae, "model"), overwrite,
                     ["test_model.gin"], gin_bindings)

# 3. Extract the mean representation for these models.
for path in [path_custom_vae]:
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    postprocess_gin = ["postprocess.gin"]  # This contains the settings.
    # postprocess.postprocess_with_gin defines the standard extraction protocol.
    postprocess.postprocess_with_gin(model_path, representation_path,
                                     overwrite, postprocess_gin)

# 4. Compute the Factor VAE metric (already implemented) for both models.
gin_bindings = [
    "evaluation.evaluation_fn = @factor_vae_score",
    "dataset.name = 'dummy_data'", "evaluation.random_seed = 0",
    "factor_vae_score.num_variance_estimate = 100",
    "factor_vae_score.num_train = 1000"
    "factor_vae_score.num_eval = 100", "factor_vae_score.batch_size = 5"
]
for path in [path_custom_vae]:
    result_path = os.path.join(path, "metrics", "fvm")
    representation_path = os.path.join(path, "representation")
    evaluate.evaluate_with_gin(representation_path,
                               result_path,
                               overwrite,
                               gin_bindings=gin_bindings)

# 5. Compute the Mutual Information Gap (already implemented) for both models.
gin_bindings = [
    "evaluation.evaluation_fn = @mig", "dataset.name='auto'",
    "evaluation.random_seed = 0", "mig.num_train=1000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]
for path in [path_custom_vae]:
    result_path = os.path.join(path, "metrics", "mig")
    representation_path = os.path.join(path, "representation")
    evaluate.evaluate_with_gin(representation_path,
                               result_path,
                               overwrite,
                               gin_bindings=gin_bindings)

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
