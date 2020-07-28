# lin_ucv
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____lin_ucb_06192a8795,TrivagoLogisticModelInteraction_selu____lin_ucb_1280167168,TrivagoLogisticModelInteraction_selu____lin_ucb_280e896d59,TrivagoLogisticModelInteraction_selu____lin_ucb_3782147584,TrivagoLogisticModelInteraction_selu____lin_ucb_568e322ba8,TrivagoLogisticModelInteraction_selu____lin_ucb_82001fac75,TrivagoLogisticModelInteraction_selu____lin_ucb_9484a88039,TrivagoLogisticModelInteraction_selu____lin_ucb_987f96b239,TrivagoLogisticModelInteraction_selu____lin_ucb_ae0d54efa6,TrivagoLogisticModelInteraction_selu____lin_ucb_b19dba8bc2,TrivagoLogisticModelInteraction_selu____lin_ucb_bf8a67151b,TrivagoLogisticModelInteraction_selu____lin_ucb_db3036d4d6,TrivagoLogisticModelInteraction_selu____lin_ucb_e9e5b77548 --legend bandit_policy,bandit_policy_params.alpha,full_refit,val_split_type --output  tools/eval_viz/export/Como_italy/lin_ucb


# custom_lin_ucb
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____custom_lin_ucb_14b9caffea,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_232137e5c3,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_5a8a492f22,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_8eddde6139,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_9044c3e788,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_9a4e700253,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_b9bb0e9981,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_c593ae9dc5,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_c809d6b3de,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_e21301c529,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_f7c5a8bd7f,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_fdc3a45d21 --legend bandit_policy,bandit_policy_params.alpha,full_refit,val_split_type --output  tools/eval_viz/export/Como_italy/custom_lin_ucb

# epsilon_greedy
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____epsilon_greedy_0ab8944570,TrivagoLogisticModelInteraction_selu____epsilon_greedy_28e1f9a94c,TrivagoLogisticModelInteraction_selu____epsilon_greedy_3512f304c8,TrivagoLogisticModelInteraction_selu____epsilon_greedy_3ba6d2f02b,TrivagoLogisticModelInteraction_selu____epsilon_greedy_3ee109563f,TrivagoLogisticModelInteraction_selu____epsilon_greedy_6e2799624c,TrivagoLogisticModelInteraction_selu____epsilon_greedy_6e7c6b83f8,TrivagoLogisticModelInteraction_selu____epsilon_greedy_7270e23121,TrivagoLogisticModelInteraction_selu____epsilon_greedy_99dc76413f,TrivagoLogisticModelInteraction_selu____epsilon_greedy_a455cb3397,TrivagoLogisticModelInteraction_selu____epsilon_greedy_c435ffba91,TrivagoLogisticModelInteraction_selu____epsilon_greedy_f326b2e55a --legend bandit_policy,bandit_policy_params.epsilon,full_refit,val_split_type --output tools/eval_viz/export/Como_italy/epsilon_greedy

# softmax
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____softmax_explorer_32ef87ed6a,TrivagoLogisticModelInteraction_selu____softmax_explorer_397afac6e4,TrivagoLogisticModelInteraction_selu____softmax_explorer_95c129e8b5,TrivagoLogisticModelInteraction_selu____softmax_explorer_bb327c6ac0,TrivagoLogisticModelInteraction_selu____softmax_explorer_d53d967be5,TrivagoLogisticModelInteraction_selu____softmax_explorer_dd733bc8cd,TrivagoLogisticModelInteraction_selu____softmax_explorer_f03038b928,TrivagoLogisticModelInteraction_selu____softmax_explorer_f600cae1c1 --legend bandit_policy,bandit_policy_params.logit_multiplier,full_refit,val_split_type --output tools/eval_viz/export/Como_italy/softmax

# lin_ts
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____lin_ts_0294d3dbac,TrivagoLogisticModelInteraction_selu____lin_ts_08699407d4,TrivagoLogisticModelInteraction_selu____lin_ts_54ed5d3eaf,TrivagoLogisticModelInteraction_selu____lin_ts_6e2647af30,TrivagoLogisticModelInteraction_selu____lin_ts_8c3544b666,TrivagoLogisticModelInteraction_selu____lin_ts_8ed94b7392,TrivagoLogisticModelInteraction_selu____lin_ts_bd34b87cda,TrivagoLogisticModelInteraction_selu____lin_ts_e2ac5fd55a --legend bandit_policy,bandit_policy_params.v_sq,full_refit,val_split_type --output tools/eval_viz/export/Como_italy/lin_ts

# percentil
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____percentile_adapt_17194c7c38,TrivagoLogisticModelInteraction_selu____percentile_adapt_4ffe9f713d,TrivagoLogisticModelInteraction_selu____percentile_adapt_5d76a63674,TrivagoLogisticModelInteraction_selu____percentile_adapt_b54aa664de,TrivagoLogisticModelInteraction_selu____percentile_adapt_c36bcb3c7a,TrivagoLogisticModelInteraction_selu____percentile_adapt_f55d1befce --legend bandit_policy,bandit_policy_params.exploration_threshold,full_refit,val_split_type --output tools/eval_viz/export/Como_italy/percentile

# adaptative
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____adaptive_15ca3b24ea,TrivagoLogisticModelInteraction_selu____adaptive_21d30e4bf5,TrivagoLogisticModelInteraction_selu____adaptive_2786776728,TrivagoLogisticModelInteraction_selu____adaptive_2f2dc3382c,TrivagoLogisticModelInteraction_selu____adaptive_456408f8b3,TrivagoLogisticModelInteraction_selu____adaptive_4a1267b871,TrivagoLogisticModelInteraction_selu____adaptive_5496de97cd,TrivagoLogisticModelInteraction_selu____adaptive_7134f5cf8c,TrivagoLogisticModelInteraction_selu____adaptive_74f7ff3544,TrivagoLogisticModelInteraction_selu____adaptive_963556ca7d,TrivagoLogisticModelInteraction_selu____adaptive_b688333a5f,TrivagoLogisticModelInteraction_selu____adaptive_ca4f001170 --legend bandit_policy,bandit_policy_params.exploration_threshold,bandit_policy_params.decay_rate,full_refit,val_split_type --output tools/eval_viz/export/Como_italy/adaptative


# explore_then_exp
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____explore_then_exp_436c532cf3,TrivagoLogisticModelInteraction_selu____explore_then_exp_587e4d681a,TrivagoLogisticModelInteraction_selu____model_3e9f9637b0 --legend bandit_policy,bandit_policy_params.explore_rounds,bandit_policy_params.decay_rate  --output tools/eval_viz/export/explore_then_exp


#geral
python tools/eval_viz/extract_plots.py --models TrivagoLogisticModelInteraction_selu____adaptive_15ca3b24ea,TrivagoLogisticModelInteraction_selu____custom_lin_ucb_c593ae9dc5,TrivagoLogisticModelInteraction_selu____epsilon_greedy_7270e23121,TrivagoLogisticModelInteraction_selu____explore_then_exp_436c532cf3,TrivagoLogisticModelInteraction_selu____lin_ts_08699407d4,TrivagoLogisticModelInteraction_selu____lin_ucb_e9e5b77548,TrivagoLogisticModelInteraction_selu____percentile_adapt_17194c7c38,TrivagoLogisticModelInteraction_selu____softmax_explorer_bb327c6ac0,TrivagoLogisticModelInteraction_selu____random_21f4fc3a3a,TrivagoLogisticModelInteraction_selu____model_3e9f9637b0 --legend bandit_policy  --output tools/eval_viz/export/geral