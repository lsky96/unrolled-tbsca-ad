import os
import argparse

import numpy as np
import torch

import config
import utils
import datagen
import datagen_rw
import paper_results_utils as putils
import unrolled_bsca
import unrolled_bsca_tensor

import torch.multiprocessing as mp

from config import DFILEEXT, SCENARIODIR, RESULTDIR, EXPORTDIR, RW_ABILENE_DIR, CPU_THREAD_LIMIT

if not os.path.isdir(SCENARIODIR):
    os.mkdir(SCENARIODIR)

if not os.path.isdir(RESULTDIR):
    os.mkdir(RESULTDIR)

if not os.path.isdir(EXPORTDIR):
    os.mkdir(EXPORTDIR)

"""CONFIGURATION: Sub-experiments"""
"""CONFIG: synthethic_adaptfeatures_model_comp"""
AFMC_LAYER_SWEEP = False
AFMC_DF_SWEEP = False  # for iteratively sweeping features for W and comparing model performance
AFMC_MU_SWEEP = False  # for iteratively sweeping features for M and comparing model performance
AFMC_NW_HIDLAYER_SWEEP = False  # sweeping number of hidden layers for statistical parameter representation map

"""CONFIG: tensor_synthetic_comparisons"""
SYNTH_LAYER_SWEEP = True
SYNTH_TDATA_SIZE_SWEEP = False
SYNTH_GRAPH_SIZE_SWEEP = False  # sweeping over datasets with different domain sizes
# GRAPH_SIZE_SENSITIVITY = True
SYNTH_LOSSFUN_SS_SWEEP = False
SYNTH_CFG_SWEEP = False  # various configurations
SYNTH_CLASSICAL = False  # classical algorithms
SYNTH_HANKEL_ALG = False  # Hankel-tensor-based algorithm (Kasai 2016)
SYNTH_RUNTIME = False  # measure runtime, turns of thread limit

"""CONFIG: tensor_rw_abilene_comparisons"""
RW_LAYER_SWEEP = True
RW_CLASSICAL = False  # classical algorithms
RW_HANKEL_REFALG = False  # Hankel-tensor-based algorithm (Kasai 2016)


def get_lam_mu_grid(min, max, resolution):
    lam_log_space = torch.linspace(min, max, resolution)
    mu_log_space = torch.linspace(min, max, resolution)
    return lam_log_space, mu_log_space


def synthetic_classical_comparison(cvs=None):
    if isinstance(CPU_THREAD_LIMIT, int):
        torch.set_num_threads(CPU_THREAD_LIMIT)
    data_set_name = "cpd_s"
    data_path = os.path.join(SCENARIODIR, data_set_name + DFILEEXT)

    if not os.path.isfile(data_path):
        num_timesteps = 200  # was 100
        num_time_seg = 10  # was 10
        graph_param = {
            "num_nodes": 10,
            "num_edges": 15,
            "min_distance": 1.0,
        }
        sampling_param = {
            "flow_distr": {"structure": "cpd",
                           "rank": 30,
                           "scale": [1.0, 1.0, "het_flows", "het_time"]},
            "anomaly_distr": {"prob": 0.005,
                              "amplitude": [0.5, "het_mirror_normal"]},
            "noise_distr": {"variance": [0.01, "het_mirror_normal"]},
            "observation_prob": 0.9,
        }

        # num_timesteps = 300  # was 100
        # num_time_seg = 10  # was 10
        # graph_param = {
        #     "num_nodes": 15,
        #     "num_edges": 30,
        #     "min_distance": 1.0,
        # }
        # sampling_param = {
        #     "flow_distr": {"structure": "cpd",
        #                    "rank": 70,
        #
        #                    "scale": [0.25, 1.0, "het_flows", "het_time"]},
        #     "anomaly_distr": {"prob": 0.005,
        #                       "amplitude": [0.8, "het_mirror_normal"]},
        #     "noise_distr": {"variance": [0.04, "het_mirror_normal"]},
        #     "observation_prob": 0.9,
        # }
        scenario_param = {
            "num_timesteps": num_timesteps,
            "num_time_seg": num_time_seg,
            "graph_param": graph_param,
            "sampling_param": sampling_param}

        data = datagen.generate_synthetic_nw_scenarios(batch_size=250, **scenario_param)
        torch.save(data, data_path)
        print("Data saved.")

    all_idx = list(range(250))
    cval_idx = [list(range(0, 50)),
                list(range(50, 100)),
                list(range(100, 150)),
                list(range(150, 200)),
                list(range(200, 250)), ]

    default_cvs = list(range(len(cval_idx)))

    if cvs is not None:
        cvs = cvs if isinstance(cvs, list) else [cvs]
        assert (set(cvs) <= set(default_cvs))
    else:
        cvs = default_cvs

    """General Model Param"""
    rank_svd = 30
    rank_cpd = 200

    ALGS = ["bsca", "bbcd", "bsca_tens_rlx_it1nrlx", "bsca_tens_nrlx"]
    RANKS = [rank_svd, rank_svd, rank_cpd, rank_cpd]

    for cv in cvs:

        val_idx = cval_idx[cv]
        train_idx = [idx for idx in all_idx if idx not in val_idx]

        train_path = {"path": data_path, "idx": train_idx}
        val_path = {"path": data_path, "idx": val_idx}
        data_set_name_cv = data_set_name + "_cv{}".format(cv)

        """Grid Search"""
        lam_log_space, mu_log_space = get_lam_mu_grid(-6, 2, 33)

        # result_name1 = "gridsearch_BSCA_100iter_r{}_on_{}".format(rank_svd, data_set_name_cv)
        # putils.gridsearch(val_path, result_name1, lam_log_space, mu_log_space, rank_svd,
        #                   inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bsca")
        # result_name2 = "gridsearch_BBCD_100iter_r{}_on_{}".format(rank_svd, data_set_name_cv)
        # putils.gridsearch(val_path, result_name2, lam_log_space, mu_log_space, rank_svd,
        #                   inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bbcd")
        # result_name3 = "gridsearch_BBCDr_100iter_r{}_on_{}".format(rank_svd, data_set_name_cv)
        # putils.gridsearch(val_path, result_name3, lam_log_space, mu_log_space, rank_svd,
        #                   inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bbcd_r")
        result_name4 = "gridsearch_BSCAtensor_nrlx_100iter_r{}_on_{}".format(rank_cpd, data_set_name_cv)
        putils.gridsearch(val_path, result_name4, lam_log_space, mu_log_space, rank_cpd,
                          inv_layers=[100], num_iter=100, init="randsc", alg="bsca_tens_nrlx")
        # result_name5 = "gridsearch_BSCAtensor_rlx_100iter_r{}_on_{}".format(rank_cpd, data_set_name_cv)
        # putils.gridsearch(val_path, result_name5, lam_log_space, mu_log_space, rank_cpd,
        #                   inv_layers=list(range(1, 101)), num_iter=100, init="randsc", alg="bsca_tens_rlx")

        # putils.show_results_gridsearch(result_name1, layers_to_show=[5, 10, 100])
        # putils.show_results_gridsearch(result_name2, layers_to_show=[5, 10, 100])
        # # putils.show_results_gridsearch(result_name3, layers_to_show=[5, 10, 100])
        putils.show_results_gridsearch(result_name4, layers_to_show=[100])
        # putils.show_results_gridsearch(result_name5, layers_to_show=[5, 10, 100])

        """Classical Algorithm Parameter Optimization"""
        for alg, rank in zip(ALGS, RANKS):
            result_opt = "bayopt_{}_r{}_on_{}".format(alg, rank, data_set_name_cv)
            putils.eval_classical_alg(train_path, val_path, result_opt, rank, num_fun_calls=400, alg=alg, auc_over_iter=True)

        # result_opt1 = "bayopt_BSCA_r{}_on_{}".format(rank_svd, data_set_name_cv)
        # putils.eval_classical_alg(train_path, val_path, result_opt1, rank_svd, num_fun_calls=250, alg="bsca")
        # result_opt2 = "bayopt_BBCD_r{}_on_{}".format(rank_svd, data_set_name_cv)
        # putils.eval_classical_alg(train_path, val_path, result_opt2, rank_svd, num_fun_calls=250, alg="bbcd")
        # result_opt3 = "bayopt_BSCAtens_nrlx_r{}_on_{}".format(rank_cpd, data_set_name_cv)
        # putils.eval_classical_alg(train_path, val_path, result_opt3, rank_cpd, num_fun_calls=250, alg="bsca_tens_nrlx")
        # result_opt4 = "bayopt_BSCAtens_rlx_r{}_on_{}".format(rank_cpd, data_set_name_cv)
        # putils.eval_classical_alg(train_path, val_path, result_opt4, rank_cpd, num_fun_calls=250, alg="bsca_tens_rlx")


def synthetic_adaptfeatures_model_comp(cvs=None):
    """
    Runs adaptive model using only certain features. Somewhat manual, loops and used features must be set below.
    """

    if isinstance(CPU_THREAD_LIMIT, int):
        torch.set_num_threads(CPU_THREAD_LIMIT)
    data_set_name = "small_cpd_het"
    data_path = os.path.join(SCENARIODIR, data_set_name + DFILEEXT)

    if not os.path.isfile(data_path):
        num_timesteps = 100
        num_time_seg = 10
        graph_param = {
            "num_nodes": 10,
            "num_edges": 25,
            "min_distance": 1.0,
        }
        sampling_param = {
            "flow_distr": {"structure": "cpd",
                           "rank": 40,
                           "scale": [0.25, 1.0, "het_flows", "het_time"]},
            "anomaly_distr": {"prob": 0.005,
                              "amplitude": [1.5, "het_mirror_normal"]},
            "noise_distr": {"variance": [0.25, "het_mirror_normal"]},
            "observation_prob": 0.95,
        }
        scenario_param = {
            "num_timesteps": num_timesteps,
            "num_time_seg": num_time_seg,
            "graph_param": graph_param,
            "sampling_param": sampling_param}

        data = datagen.generate_synthetic_nw_scenarios(batch_size=500, **scenario_param)
        torch.save(data, data_path)
        print("Data saved.")

    all_idx = list(range(500))
    cval_idx = [list(range(0, 100)),
                list(range(100, 200)),
                list(range(200, 300)),
                list(range(300, 400)),
                list(range(400, 500)), ]
    default_cvs = list(range(len(cval_idx)))

    if cvs is not None:
        cvs = cvs if isinstance(cvs, list) else [cvs]
        assert (set(cvs) <= set(default_cvs))
    else:
        cvs = default_cvs

    """General Model Param"""
    rank_svd = 50
    rank_cpd = 100

    num_epochs = 600
    batch_size = 10
    loss_type = "approxauc_homotopy_ss16"
    loss_options = {"beta1": 10, "beta2": 100, "t1": 150, "t2": 350}
    opt_kw = {"lr": 0.01, "weight_decay": 0.01, "betas": [0.9, 0.95]}
    sched_kw = {"milestones": [100, 500, 550, 580, 590], "gamma": 0.25, "warmup_num_steps": 5}  # warmup 20
    train_param = {"num_epochs": num_epochs, "batch_size": batch_size,
                   "opt_kw": opt_kw, "sched_kw": sched_kw,
                   "loss_type": loss_type, "loss_options": loss_options}

    ORDER = 1
    add_acr = "_test"

    for cv in cvs[::ORDER]:

        val_idx = cval_idx[cv]
        train_idx = [idx for idx in all_idx if idx not in val_idx]

        train_data = {"path": data_path, "idx": train_idx}
        val_data = {"path": data_path, "idx": val_idx}

        cvacr = "cv{}".format(cv)

        if AFMC_LAYER_SWEEP:
            for num_layers in [3, 4, 5, 6, 7, 8, 9, 10]:
                """TENSOR"""
                df_fmask = torch.zeros(8, dtype=torch.float64)
                mu_fmask = torch.zeros(32, dtype=torch.float64)

                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": False,  # "modewise",
                             # "datafit_options": {"1dyn1ly": None, "fmask": df_fmask},
                             # "mu_options": {"d1ly": None, "fmask": mu_fmask},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

        """Set this according to the results of the sweep"""
        NUM_LAYERS_BEST = 6

        if AFMC_DF_SWEEP:
            num_layers = NUM_LAYERS_BEST

            """DATAFIT: LOOP CONFIG (additional features)"""
            ### Round 0
            # for feat in [0, 1, 2, 3, 4][::ORDER]

            ### Round 1
            # for feat in [2, 3, 4][::ORDER]:

            ### Round 2
            for feat in [2, 4][::ORDER]:

                ### Masks
                df_fmask = torch.zeros(8, dtype=torch.float64)
                mu_fmask = torch.zeros(32, dtype=torch.float64)


                """DATAFIT: SET FEATURES"""
                ### from Round 0
                df_feat_name = "ftdf"

                ### from Round 1
                df_fmask[[0, 2, 4]] = 1.0  # feat1
                df_feat_name = "ftdf1"

                ### from Round 2
                df_fmask[6] = 1.0  # feat3
                df_feat_name = "ftdf13"

                """DATAFIT: OPTIONS"""
                if feat == 0:
                    pass
                elif feat == 1:
                    df_fmask[[0, 2, 4]] = 1.0
                elif feat == 2:
                    df_fmask[[1, 3, 6]] = 1.0
                elif feat == 3:
                    df_fmask[6] = 1.0
                elif feat == 4:
                    df_fmask[7] = 1.0

                acr = cvacr + "_{}{}".format(df_feat_name, feat) + add_acr

                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "datafit_options": {"1dyn1ly": None, "fmask": df_fmask},
                             # "mu_options": {"d1ly": None, "fmask": mu_fmask},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=acr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

        if AFMC_MU_SWEEP:
            num_layers = NUM_LAYERS_BEST

            """MU: LOOP CONFIG (additional features)"""
            # ### Round 0
            # for feat in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][::ORDER]:

            # ### Round 1
            # for feat in [1, 2, 4, 5, 6, 7, 8, 9][::ORDER]:

            # ### Round 2
            # for feat in [1, 2, 4, 5, 6, 7, 9][::ORDER]:

            # ### Round 3
            # for feat in [1, 2, 4, 5, 7, 9][::ORDER]:

            # ### Round 3
            # for feat in [1, 2, 4, 7, 9][::ORDER]:

            ### Round 4
            for feat in [1, 2, 4, 9][::ORDER]:

                ### Masks
                df_fmask = torch.zeros(8, dtype=torch.float64)
                mu_fmask = torch.zeros(32, dtype=torch.float64)

                """MU: SET FEATURES"""
                # ### from Round 0
                mu_feat_name = "ftmu"

                # ### from Round 1
                mu_fmask[[2, 12, 22]] = 1.0  # feat3 # Avar
                mu_feat_name = "ftmu3"

                # ### from Round 2
                mu_fmask[30] = 1.0  # feat8 # obs
                mu_feat_name = "ftmu38"

                # ### from Round 3
                mu_fmask[[5, 15, 25]] = 1.0  # feat6 # proj_err_max
                mu_feat_name = "ftmu386"

                # ### from Round 3
                mu_fmask[[4, 14, 24]] = 1.0  # feat5 # sca_Amax
                mu_feat_name = "ftmu3865"

                # ### from Round 4
                mu_fmask[[6, 16, 26]] = 1.0  # feat7 # scpe_proj_err_max
                mu_feat_name = "ftmu38657"

                """MU: OPTIONS"""
                if feat == 0:
                    pass
                elif feat == 1:
                    mu_fmask[[0, 10, 20]] = 1.0
                elif feat == 2:
                    mu_fmask[[1, 11, 21]] = 1.0
                elif feat == 3:
                    mu_fmask[[2, 12, 22]] = 1.0
                elif feat == 4:
                    mu_fmask[[3, 13, 23]] = 1.0
                elif feat == 5:
                    mu_fmask[[4, 14, 24]] = 1.0
                elif feat == 6:
                    mu_fmask[[5, 15, 25]] = 1.0
                elif feat == 7:
                    mu_fmask[[6, 16, 26]] = 1.0
                elif feat == 8:
                    mu_fmask[30] = 1.0
                elif feat == 9:
                    mu_fmask[31] = 1.0

                acr = cvacr + "_{}{}".format(mu_feat_name, feat) + add_acr

                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             # "datafit_options": {"1dyn1ly": None, "fmask": df_fmask},
                             "mu_options": {"d1ly": None, "fmask": mu_fmask},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=acr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

        if AFMC_NW_HIDLAYER_SWEEP:

            for num_hidden_layers in [0, 1][::ORDER]:
                """TENSOR"""
                num_layers = NUM_LAYERS_BEST
                acr = cvacr + add_acr

                pnw_option = "f{}ly".format(num_hidden_layers + 1)

                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "datafit_options": {pnw_option: None},
                             "mu_options": {pnw_option: None},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=acr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)


def synthetic_comparisons(cvs=None):

    if (not SYNTH_RUNTIME) and isinstance(CPU_THREAD_LIMIT, int):
        torch.set_num_threads(CPU_THREAD_LIMIT)

    ORDER = 1

    """BASE DATASET"""
    data_set_name = "cpd_h"  # var 0.1, obs0.95
    data_path = os.path.join(SCENARIODIR, data_set_name + DFILEEXT)

    if not os.path.isfile(data_path):
        num_timesteps = 300
        num_time_seg = 10
        graph_param = {
            "num_nodes": 15,
            "num_edges": 30,
            "min_distance": 1.0,
        }
        sampling_param = {
            "flow_distr": {"structure": "cpd",
                           "rank": 70,
                           "scale": [0.25, 1.0, "het_flows", "het_time"]},
            "anomaly_distr": {"prob": 0.005,
                              "amplitude": [0.8, "het_mirror_normal"]},
            "noise_distr": {"variance": [0.04, "het_mirror_normal"]},
            "observation_prob": 0.90,
        }
        scenario_param = {
            "num_timesteps": num_timesteps,
            "num_time_seg": num_time_seg,
            "graph_param": graph_param,
            "sampling_param": sampling_param}

        data = datagen.generate_synthetic_nw_scenarios(batch_size=500, **scenario_param)
        torch.save(data, data_path)
        print("Data saved.")

    """General Param"""
    rank_svd = 60  # min(2*E, T)
    rank_cpd = 300  # min(2*E*T1, 2*E*T2, T=T1*T2)

    # num_epochs = 600
    num_epochs = 500
    batch_size = 10
    loss_type = "approxauc_homotopy_ss16"
    loss_options = {"beta1": 10, "beta2": 100, "t1": 125, "t2": 275}
    opt_kw = {"lr": 0.01, "weight_decay": 0.05, "betas": [0.9, 0.95]}
    sched_kw = {"milestones": [100, 400, 450, 480, 490], "gamma": 0.25, "warmup_num_steps": 5}
    train_param = {"num_epochs": num_epochs, "batch_size": batch_size,
                   "opt_kw": opt_kw, "sched_kw": sched_kw, "weight_decay": {0: 0.05, 350: 0.01},
                   "loss_type": loss_type, "loss_options": loss_options}
    # loss_options = {"beta1": 10, "beta2": 100, "t1": 100, "t2": 200}
    # opt_kw = {"lr": 0.01, "weight_decay": 0.05, "betas": [0.9, 0.95]}
    # sched_kw = {"milestones": [75, 225, 275, 290, 295], "gamma": 0.25, "warmup_num_steps": 5}
    # train_param = {"num_epochs": num_epochs, "batch_size": batch_size,
    #                "opt_kw": opt_kw, "sched_kw": sched_kw, "weight_decay": {0: 0.05, 205: 0.01},
    #                "loss_type": loss_type, "loss_options": loss_options}

    all_idx = list(range(500))
    cval_idx = [list(range(0, 100)),
                list(range(100, 200)),
                list(range(200, 300)),
                list(range(300, 400)),
                list(range(400, 500)), ]

    default_cvs = list(range(len(cval_idx)))

    if cvs is not None:
        cvs = cvs if isinstance(cvs, list) else [cvs]
        assert (set(cvs) <= set(default_cvs))
    else:
        cvs = default_cvs

    for cv in cvs[::ORDER]:

        val_idx = cval_idx[cv]
        train_idx = [idx for idx in all_idx if idx not in val_idx]

        roll_idx = all_idx.index(val_idx[0])
        train_idx = np.roll(np.array(train_idx), -roll_idx).tolist()  # rolling indices such that list starts at
        # different index each time, important when reducing data size

        train_data = {"path": data_path, "idx": train_idx}
        val_data = {"path": data_path, "idx": val_idx}

        cvacr = "cv{}".format(cv)
        pwn_opt = "f1ly"

        if SYNTH_LAYER_SWEEP:
            for num_layers in range(3, 10):
                """TENSOR with features"""
                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "datafit_options": {pwn_opt: None},
                             "mu_options": {pwn_opt: None},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

            for num_layers in range(3, 12):
                """Matrix-Factor with features"""
                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_svd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "no_tensor": True,
                             "datafit_options": {pwn_opt: None},
                             "mu_options": {pwn_opt: None},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

            """CAMSAP"""
            for num_layers in range(3, 9):
                model = unrolled_bsca.BSCAUnrolled
                model_param = {"nn_model_class": model,
                               "num_layers": num_layers, "rank": rank_svd,
                               "param_nw": True,
                               }

                model_name = putils.get_run_name(data_set_name, **model_param, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **model_param,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, model, val_data,
                                  auc_over_layers=True, training_stats=True)

            for num_layers in range(5, 11):
                """TENSOR non-adaptive"""
                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

        if SYNTH_CLASSICAL:
            ALGS = ["bbcd", "bsca_tens_rlx_it1nrlx"]
            RANKS = [rank_svd, rank_cpd]
            # ALGS = ["bbcd"]
            # RANKS = [rank_svd]

            """Classical Algorithm Parameter Optimization"""
            train_idx_reduced = train_data["idx"][:100]  # otherwise, it will take way too long, this should not affect the result too much since overfitting is impossible
            train_data_reduced = {"path": train_data["path"], "idx": train_idx_reduced}
            for alg, rank in zip(ALGS, RANKS):
                result_name = "bayopt_{}_r{}_on_{}".format(alg, rank, data_set_name + "_" + cvacr)
                putils.eval_classical_alg(train_data_reduced, val_data, result_name, rank, num_fun_calls=400, alg=alg,
                                          auc_over_iter=False)

        if SYNTH_HANKEL_ALG:
            window_length_kasai = None
            rank_kasai = rank_cpd

            if window_length_kasai is None:
                wlk_str = "x"
                search_space = [(-4.0, 4.0),
                                (-8.0, -2.0),
                                (-8.0, 0.0),
                                (-6.0, 0.0),
                                (2, 30)]  # large windows were tried and performed significantly worse here
            else:
                wlk_str = window_length_kasai
                search_space = [(-4.0, 4.0),
                                (-8.0, -2.0),
                                (-8.0, 0.0),
                                (-6.0, 0.0)]

            result_name = "refalg_kasai_r{}_w{}_ON_{}".format(rank_kasai, wlk_str, data_set_name) \
                          + "_" + cvacr

            train_idx_reduced = train_data["idx"][:25]  # otherwise, it will take way too long, this should not affect the result too much since overfitting is impossible
            train_data_reduced = {"path": train_data["path"], "idx": train_idx_reduced}
            putils.eval_kasai_refalg(train_data_reduced, val_data, result_name, rank_kasai, window_length_kasai,
                                     batch_partition_size=50, search_space=search_space, num_fun_calls=400)

        # Fixed number of layers from here on out
        num_layers_svdmw = 10  # best
        num_layers_tensmw = 8  # best
        num_layers_svdcamsap = 7  # best

        if SYNTH_TDATA_SIZE_SWEEP:
            for tset_size in [10, 25, 50, 100, 200, 400]:
                train_idx_reduced = train_data["idx"][:tset_size]
                train_data_reduced = {"path": train_data["path"], "idx": train_idx_reduced}

                acr = cvacr + "_tsetsz{}".format(tset_size)

                """TENSOR with features"""
                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers_tensmw, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "datafit_options": {pwn_opt: None},
                             "mu_options": {pwn_opt: None},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param, acronym=acr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data_reduced, val_data, **train_param, **param_cpd,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

        if SYNTH_GRAPH_SIZE_SWEEP:
            szs = [0, 1, 2]

            """Data Generation"""
            for sz in szs:
                data_set_name_sz = "cpd_h_sz{}".format(sz)
                data_path_sz = os.path.join(SCENARIODIR, data_set_name_sz + DFILEEXT)

                if not os.path.isfile(data_path_sz):
                    num_timesteps = 300
                    num_time_seg = 10

                    graph_param = {  # ratio: 2 =~ num_nodes**1.5 / num_edges
                        "num_nodes": [8, 16, 32][sz],
                        "num_edges": [12, 32, 90][sz],
                        "min_distance": 1.0,
                    }

                    sampling_param = {
                        "flow_distr": {"structure": "cpd",
                                       "rank": 70,
                                       "scale": [0.25, 1.0, "het_flows", "het_time"]},
                        "anomaly_distr": {"prob": 0.005,
                                          "amplitude": [0.8, "het_mirror_normal"]},
                        "noise_distr": {"variance": [0.04, "het_mirror_normal"]},
                        "observation_prob": 0.90,
                    }

                    scenario_param = {
                        "num_timesteps": num_timesteps,
                        "num_time_seg": num_time_seg,
                        "graph_param": graph_param,
                        "sampling_param": sampling_param}

                    data = datagen.generate_synthetic_nw_scenarios(batch_size=500, **scenario_param)
                    torch.save(data, data_path_sz)
                    print("Data saved.")

            """Runs"""
            for sz in szs:
                data_set_name_sz = "cpd_h_sz{}".format(sz)
                data_path_sz = os.path.join(SCENARIODIR, data_set_name_sz + DFILEEXT)

                train_data_sz = {"path": data_path_sz, "idx": train_data["idx"]}
                val_data_sz = {"path": data_path_sz, "idx": val_data["idx"]}

                """TENSOR with features"""
                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers_tensmw, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "datafit_options": {pwn_opt: None},
                             "mu_options": {pwn_opt: None},
                             }
                model_name = putils.get_run_name(data_set_name_sz, **param_cpd, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data_sz, val_data_sz, **train_param, **param_cpd,
                                   batch_partition_size=({"train": 5, "val": 25} if sz == 2 else None))  # memory optimization
                if True:  # TODO remove
                    putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data_sz,
                                      auc_over_layers=True, training_stats=True)

                """Adaptation"""
                for sz_data in szs:
                    if sz_data != sz:
                        data_set_name_sz = "cpd_h_sz{}".format(sz_data)
                        data_path_sz = os.path.join(SCENARIODIR, data_set_name_sz + DFILEEXT)
                        val_data_sz = {"path": data_path_sz, "idx": val_data["idx"]}

                        putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data_sz,
                                          auc_over_layers=True, training_stats=True)

            # if GRAPH_SIZE_SENSITIVITY:
            #     for sz in szs:
            #         data_set_name_sz = "cpd_h_sz{}".format(sz)
            #         data_path_sz = os.path.join(SCENARIODIR, data_set_name_sz + DFILEEXT)
            #         val_path = {"path": data_path_sz, "idx": (val_idx if sz != 2 else val_idx[:25])}  # reduction for memory reasons
            #         data_set_name_cv = data_set_name_sz + "_cv{}".format(cv)
            #
            #         """Grid Search"""
            #         lam_log_space, mu_log_space = get_lam_mu_grid(-5, 1, 25)
            #         result_name4 = "gridsearch_BSCAtensor_nrlx_10iter_r{}_on_{}".format(rank_cpd, data_set_name_cv)
            #         putils.gridsearch(val_path, result_name4, lam_log_space, mu_log_space, rank_cpd,
            #                           inv_layers=[25], num_iter=25, init="randsc", alg="bsca_tens_nrlx")
            #
            #         putils.show_results_gridsearch(result_name4, layers_to_show=[10])

        if SYNTH_LOSSFUN_SS_SWEEP:
            for lfss in [1, 2, 4, 8, 32]:  # 16 is default
                loss_type_temp = "approxauc_homotopy_ss"
                loss_options_temp = loss_options.copy()
                loss_options_temp["subsampling"] = lfss
                train_param_temp = {"num_epochs": num_epochs, "batch_size": batch_size,
                                    "opt_kw": opt_kw, "sched_kw": sched_kw,
                                    "loss_type": loss_type_temp, "loss_options": loss_options_temp}

                acr = cvacr + "_lfss{}".format(lfss)

                """TENSOR with features"""
                param_cpd = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                             "num_layers": num_layers_tensmw, "rank": rank_cpd,
                             "option": "rlx2x+nnf", "it1_option": "nrlx",
                             "param_nw": "modewise",
                             "datafit_options": {pwn_opt: None},
                             "mu_options": {pwn_opt: None},
                             }
                model_name = putils.get_run_name(data_set_name, **param_cpd, **train_param_temp, acronym=acr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param_temp, **param_cpd,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, unrolled_bsca_tensor.BSCATensorUnrolled, val_data,
                                  auc_over_layers=True, training_stats=True)

        if SYNTH_CFG_SWEEP:
            for cfg in [1, 2, 6, 7, 9, 10, 11, 12, 13]:
            # for cfg in [2, 9, 12]:
                # for cfg in [6]:

                """TENSOR with features"""
                # if cfg == 0:
                #     # modewise, no nrlx init
                #     param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                #                       "num_layers": num_layers_tensmw, "rank": rank_cpd,
                #                       "option": "rlx2x+nnf",
                #                       "param_nw": "modewise",
                #                       "datafit_options": {pwn_opt: None},
                #                       "mu_options": {pwn_opt: None},
                #                       }
                if cfg == 1:
                    # no modewise parameters
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": num_layers_tensmw, "rank": rank_cpd,
                                      "option": "rlx2x+nnf", "it1_option": "nrlx",
                                      }
                elif cfg == 2:
                    # no modewise, matriox
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": (num_layers_svdmw if not SYNTH_RUNTIME else num_layers_tensmw), "rank": rank_svd,
                                      "option": "rlx2x+nnf", "it1_option": "nrlx",
                                      "no_tensor": True,
                                      }
                # elif cfg == 3:
                #     # Basically not unrolled, vanilla algorithm
                #     param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                #                       "num_layers": num_layers_tensmw, "rank": rank_cpd,
                #                       "option": "rlx2x+nnf", "it1_option": "nrlx",
                #                       "shared_weights": True,
                #                       }
                # elif cfg == 4:
                #     # Basically not unrolled, vanilla algorithm, no nrlx init
                #     param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                #                       "num_layers": num_layers_tensmw, "rank": rank_cpd,
                #                       "option": "rlx2x+nnf",
                #                       "shared_weights": True,
                #                       }
                # elif cfg == 5:
                #     # No relaxation, not unrolled
                #     param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                #                       "num_layers": num_layers_tensmw, "rank": rank_cpd,
                #                       "option": "nrlx",
                #                       "shared_weights": True,
                #                       }
                elif cfg == 6:
                    # Modewise, no relaxation
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": num_layers_tensmw, "rank": rank_cpd,
                                      "option": "nrlx",
                                      "param_nw": "modewise",
                                      "datafit_options": {pwn_opt: None},
                                      "mu_options": {pwn_opt: None},
                                      }
                elif cfg == 7:
                    # No double X update in RLX, it1 nrlx
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": num_layers_tensmw, "rank": rank_cpd,
                                      "option": "rlx+nnf", "it1_option": "nrlx",
                                      "param_nw": "modewise",
                                      "datafit_options": {pwn_opt: None},
                                      "mu_options": {pwn_opt: None},
                                      }
                # elif cfg == 8:
                #     # No double X update in RLX, rlx it1
                #     param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                #                       "num_layers": num_layers_tensmw, "rank": rank_cpd,
                #                       "option": "rlx+nnf",
                #                       "param_nw": "modewise",
                #                       "datafit_options": {pwn_opt: None},
                #                       "mu_options": {pwn_opt: None},
                #                       }

                elif cfg == 9:
                    # modewise tensor, only for computation time
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": (num_layers_svdmw if not SYNTH_RUNTIME else num_layers_tensmw), "rank": rank_svd,
                                      "option": "rlx2x+nnf", "it1_option": "nrlx",
                                      "param_nw": "modewise",
                                      "no_tensor": True,
                                      "datafit_options": {pwn_opt: None},
                                      "mu_options": {pwn_opt: None},
                                      }
                elif cfg == 10:
                    param_cpd_temp = {"nn_model_class": unrolled_bsca.BSCAUnrolled,
                                        "num_layers": (num_layers_svdcamsap if not SYNTH_RUNTIME else num_layers_tensmw), "rank": rank_svd,
                                        "param_nw": True,
                                      }
                elif cfg == 11:
                    # modewise sw, tensor
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": num_layers_tensmw, "rank": rank_cpd,
                                      "option": "rlx2x+nnf", "it1_option": "nrlx",
                                      "param_nw": "modewise_sw",
                                      "datafit_options": {pwn_opt: None},
                                      "mu_options": {pwn_opt: None},
                                      }
                elif cfg == 12:
                    # modewise sw, matrix
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": num_layers_svdmw, "rank": rank_svd,
                                      "option": "rlx2x+nnf", "it1_option": "nrlx",
                                      "param_nw": "modewise_sw",
                                      "datafit_options": {pwn_opt: None},
                                      "mu_options": {pwn_opt: None},
                                      "no_tensor": True,
                                      }
                elif cfg == 13:
                    # modewise tensor, only for computation time
                    param_cpd_temp = {"nn_model_class": unrolled_bsca_tensor.BSCATensorUnrolled,
                                      "num_layers": num_layers_tensmw, "rank": rank_cpd,
                                      "option": "rlx2x+nnf", "it1_option": "nrlx",
                                      "param_nw": "modewise",
                                      "datafit_options": {pwn_opt: None},
                                      "mu_options": {pwn_opt: None},
                                      }
                else:
                    raise ValueError

                if cfg not in [9, 10, 13]:  # cfg acr can be removed, configurations only here for computation time
                    acr = cvacr + "_cfg{}".format(cfg)
                else:
                    acr = cvacr

                model_name = putils.get_run_name(data_set_name, **param_cpd_temp, **train_param, acronym=acr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **param_cpd_temp,
                                   batch_partition_size=({"train": 5, "val": 50} if cfg == 6 else None))  # memory optimization
                putils.eval_model(run_path, param_cpd_temp["nn_model_class"], val_data,
                                  auc_over_layers=True, training_stats=True, computetime=SYNTH_RUNTIME)


def rw_abilene_comparisons(cvs=None):
    if isinstance(CPU_THREAD_LIMIT, int):
        torch.set_num_threads(CPU_THREAD_LIMIT)

    num_weeks_abilene = 24

    """DATA GENERATION"""
    combine_weeks = 2
    ano_amplitude = .5
    # ano_amplitude = 1.0
    obs_prob = 0.95
    preprocessing = False  # when false, raw flows are used

    data_set_name_abilene = "abilene15_cb{}_raw_mxmul{}".format(combine_weeks, ano_amplitude)
    data_path_abilene = os.path.join(SCENARIODIR, data_set_name_abilene + DFILEEXT)

    if not os.path.isfile(data_path_abilene):
        sampling_param = {"anomaly_distr": {"amplitude": ano_amplitude, "prob": 0.005, "len": 1},
                          "observation_prob": obs_prob,
                          "anomaly_mixture": "maxmul", "subsampling": 3, "combine_weeks": combine_weeks}

        abilene_routing_path = os.path.join(RW_ABILENE_DIR, "A")
        abilene_flow_paths = [os.path.join(RW_ABILENE_DIR, "X{:02d}.gz".format(i)) for i in
                              range(1, num_weeks_abilene + 1)]

        utils.set_rng_seed(0)
        data_abilene = datagen_rw.abilene_dataset(abilene_routing_path, abilene_flow_paths,
                                                  sampling_param, preprocessing=preprocessing)
        torch.save(data_abilene, data_path_abilene)

    """General Param"""
    rank_svd = 30
    rank_cpd = 420  # 210
    # num_layers = 7

    # num_epochs = 500
    num_epochs = 600  # forcing 40 updates per epoch, thus an epoch is not really an epoch
    batch_size = 3  # max(3 // combine_weeks, 2)
    loss_type = "approxauc_homotopy_ss16"
    loss_options = {"beta1": 10, "beta2": 100, "t1": 150, "t2": 350}
    # opt_kw = {"lr": 0.01, "weight_decay": 0.01, "betas": [0.9, 0.95]}
    opt_kw = {"lr": 0.005, "weight_decay": 0.05, "betas": [0.9, 0.95]}
    sched_kw = {"milestones": [100, 500, 550, 580, 590], "gamma": 0.25, "warmup_num_steps": 5}
    train_param = {"num_epochs": num_epochs, "batch_size": batch_size,
                   "opt_kw": opt_kw, "sched_kw": sched_kw, "weight_decay": {0: 0.05, 450: 0.01},
                   "loss_type": loss_type, "loss_options": loss_options}

    all_idx = list(range(num_weeks_abilene // combine_weeks))
    cval_idx = [list(range(0, len(all_idx), 4)),
                list(range(1, len(all_idx), 4)),
                list(range(2, len(all_idx), 4)),
                list(range(3, len(all_idx), 4)), ]
    default_cvs = list(range(len(cval_idx)))

    if cvs is not None:
        cvs = cvs if isinstance(cvs, list) else [cvs]
        assert (set(cvs) <= set(default_cvs))
    else:
        cvs = default_cvs

    for cv in cvs:

        val_idx = cval_idx[cv]
        train_idx = [idx for idx in all_idx if idx not in val_idx]

        roll_idx = all_idx.index(val_idx[0])
        train_idx = np.roll(np.array(train_idx), -roll_idx).tolist()  # rolling indices such that list starts at
        # different index each time, important when reducing data size

        train_data = {"path": data_path_abilene, "idx": train_idx}
        val_data = {"path": data_path_abilene, "idx": val_idx}
        cvacr = "cv{}".format(cv)

        pwn_opt = "f1ly"

        if RW_LAYER_SWEEP:
            ## CAMSAP
            for num_lay in range(6, 10):
                model = unrolled_bsca.BSCAUnrolled
                model_param = {"nn_model_class": model,
                               "num_layers": num_lay, "rank": rank_svd,
                               "param_nw": True,
                               }

                model_name = putils.get_run_name(data_set_name_abilene, **model_param, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **model_param,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, model, val_data,
                                  auc_over_layers=True, training_stats=True, roc=(num_lay == 7))


            # Modewise Tensor
            for num_lay in range(7, 10):
                model = unrolled_bsca_tensor.BSCATensorUnrolled
                model_param = {"nn_model_class": model,
                               "num_layers": num_lay, "rank": rank_cpd,
                               "option": "rlx2x+nnf", "it1_option": "nrlx",
                               "param_nw": "modewise",
                               "datafit_options": {pwn_opt: None},
                               "mu_options": {pwn_opt: None},
                               }

                model_name = putils.get_run_name(data_set_name_abilene, **model_param, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **model_param,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, model, val_data,
                                  auc_over_layers=True, training_stats=True, roc=(num_lay == 9))

            # Modewise Matrix
            for num_lay in range(7, 10):
                model = unrolled_bsca_tensor.BSCATensorUnrolled
                model_param = {"nn_model_class": model,
                               "num_layers": num_lay, "rank": rank_svd,
                               "option": "rlx2x+nnf", "it1_option": "nrlx",
                               "param_nw": "modewise",
                               "datafit_options": {pwn_opt: None},
                               "mu_options": {pwn_opt: None},
                               "no_tensor": True,
                               }

                model_name = putils.get_run_name(data_set_name_abilene, **model_param, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **model_param,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, model, val_data,
                                  auc_over_layers=True, training_stats=True, roc=(num_lay == 7))

            # Modewise SW Tensor
            for num_lay in range(8, 9):
                model = unrolled_bsca_tensor.BSCATensorUnrolled
                model_param = {"nn_model_class": model,
                               "num_layers": num_lay, "rank": rank_cpd,
                               "option": "rlx2x+nnf", "it1_option": "nrlx",
                               "param_nw": "modewise_sw",
                               "datafit_options": {pwn_opt: None},
                               "mu_options": {pwn_opt: None},
                               }

                model_name = putils.get_run_name(data_set_name_abilene, **model_param, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **model_param,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, model, val_data,
                                  auc_over_layers=True, training_stats=True, roc=(num_lay == 8))

            # Modewise SW Matrix
            for num_lay in range(8, 9):
                model = unrolled_bsca_tensor.BSCATensorUnrolled
                model_param = {"nn_model_class": model,
                               "num_layers": num_lay, "rank": rank_svd,
                               "option": "rlx2x+nnf", "it1_option": "nrlx",
                               "param_nw": "modewise_sw",
                               "datafit_options": {pwn_opt: None},
                               "mu_options": {pwn_opt: None},
                               "no_tensor": True,
                               }

                model_name = putils.get_run_name(data_set_name_abilene, **model_param, **train_param, acronym=cvacr)
                run_path = os.path.join(RESULTDIR, model_name + DFILEEXT)

                putils.do_training(run_path, train_data, val_data, **train_param, **model_param,
                                   force_num_steps_per_epoch=40)
                putils.eval_model(run_path, model, val_data,
                                  auc_over_layers=True, training_stats=True, roc=(num_lay == 8))

        if RW_CLASSICAL:
            ALGS = ["bbcd"]
            RANKS = [rank_svd]

            """Classical Algorithm Parameter Optimization"""
            for alg, rank in zip(ALGS, RANKS):
                result_name = "bayopt_{}_r{}_on_{}".format(alg, rank, data_set_name_abilene + "_" + cvacr)
                putils.eval_classical_alg(train_data, val_data, result_name, rank, num_fun_calls=400, alg=alg,
                                          auc_over_iter=False, roc=True)

        if RW_HANKEL_REFALG:
            # rank_kasai = 60 # rank_cpd // 3
            # rank_kasai = 30 # rank_cpd
            # window_length_kasai = 288

            window_length_kasai = None
            rank_kasai = rank_cpd

            if window_length_kasai is None:
                wlk_str = "x"
            else:
                wlk_str = window_length_kasai
            result_name = "refalg_kasai_r{}_w{}_ON_{}".format(rank_kasai, wlk_str, data_set_name_abilene) \
                          + "_" + cvacr

            # search_space = [(-5.0, 5.0),
            #          (-8.0, 2.0),
            #          (-8.0, 2.0),
            #          (-8.0, 2.0),
            #          (288, 288)]
            #
            # putils.eval_kasai_refalg(train_data, val_data, result_name, rank_kasai, batch_partition_size=6, search_space=search_space)

            search_space = [(-2.0, 2.0),
                            (-6.0, -2.0),
                            (-4.0, 0.0),
                            (-3.0, 0.0),
                            (2, 5)]

            putils.eval_kasai_refalg(train_data, val_data, result_name, rank_kasai, window_length_kasai,
                                     batch_partition_size=3, search_space=search_space, num_fun_calls=200, roc=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parses experiment configuration.")
    parser.add_argument("exp", choices=["synth_abl", "synth_comp", "abil_comp", "class_comp"], action="store", help="Series of experiments. Subexperiments need to be configured inside the file.")
    parser.add_argument("--cvs", nargs="+", default=None, dest="cvs", type=int, action="store", help="Cross-validation splits to be run. ")
    parser.add_argument("--parallel", "-p", action="store_true", help="Optional, runs cross-validation experiments in parallel. Only CPU.")
    args = parser.parse_args()

    if args.exp == "synth_abl":
        experiment = synthetic_adaptfeatures_model_comp
    elif args.exp == "synth_comp":
        experiment = synthetic_comparisons
    elif args.exp == "abil_comp":
        experiment = rw_abilene_comparisons
    elif args.exp == "class_comp":
        experiment = synthetic_classical_comparison
    else:
        raise ValueError

    if args.parallel and args.cvs:
        if not config.DEVICE == torch.device("cpu"):
            raise ValueError("Cannot run parallel experiments on GPU.")
        print("Starting parallel processes for experiments.")
        with mp.Pool(5) as pool:
            pool.map(experiment, args.cvs)
        print("Finished experiments with cvs {}.".format(args.cvs))
        # for cv in args.cvs:
        #     p = mp.Process(target=experiment)
        #     p.start()
        #     print("Started experiment with cross-val index {}".format(cv))
    else:
        experiment(cvs=args.cvs)


