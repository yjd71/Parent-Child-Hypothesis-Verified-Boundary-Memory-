# Prototype-guided branch settings.
# Keep the base config backward compatible: enable the prototype branch only in a run config.

# master switch
prototype_enable = False

# feature source
prototype_feature_level = "p3"
prototype_source_branch = "student"
prototype_feature_detach_for_bank = True

# dynamic bank
prototype_bank_policy = "per_image_masked_pool_dynamic"
prototype_bank_clear_each_epoch = True
prototype_bank_rebuild_interval = 1
prototype_min_pixels = 16
prototype_bank_allow_empty = True

# retrieval
prototype_similarity = "cosine"
prototype_query_normalize = True
prototype_proto_normalize = True
prototype_topk = 16
prototype_sim_temperature = 0.05

# interaction: fu -> M(x) -> H(x)
prototype_tau = 0.07
prototype_theta_method = "kde_min"
prototype_theta_fallback = 0.0
prototype_alpha_hidden_ratio = 4

# fusion with learnable global scalar mu
prototype_mu_init = 0.5
prototype_mu_min = 0.0
prototype_mu_max = 1.0
prototype_mu_lr = 1e-3
prototype_mu_weight_decay = 0.0
prototype_mu_labeled_only = True
prototype_mu_detach_unlabeled = True

# supervision weights
prototype_loss_weight_h = 0.3
prototype_unsup_loss_weight = 0.1
prototype_warmup_build_only = True

# runtime behavior
prototype_checkpoint_policy = "save_and_load"
prototype_eval_policy = "checkpoint_then_rebuild"
prototype_use_in_inference = True
prototype_use_in_evaluator = True
