import os
import time
import numpy as np
import torch
import utils
import datagen
import evalfun
import reference_algo
import unrolled_bsca, unrolled_bsca_tensor
import training_tensor as training
import matplotlib.pyplot as plt
# from warnings import deprecated

from config import DFILEEXT, SCENARIODIR, EXPORTDIR, RESULTDIR, DEVICE, FP_DTYPE


def generate_data(name, dir, training_size=0, test_size=0, num_timesteps=None, num_time_seg=1, graph_param=None,
                  sampling_param=None, nflow_version="exp+exp"):
    utils.set_rng_seed(0)
    training_data_path = os.path.join(dir, name + "_training.pt")
    test_data_path = os.path.join(dir, name + "_test.pt")

    if training_size > 0 and (not os.path.isfile(training_data_path)):
        training_data = datagen.generate_synthetic_nw_scenarios(batch_size=training_size, num_timesteps=num_timesteps,
                                                                num_time_seg=num_time_seg, graph_param=graph_param,
                                                                sampling_param=sampling_param,
                                                                nflow_version=nflow_version)

        torch.save(training_data, training_data_path)
        print("Training data saved.")
    else:
        print("Training data already exists.")

    if test_size > 0 and (not os.path.isfile(test_data_path)):
        test_data = datagen.generate_synthetic_nw_scenarios(batch_size=test_size, num_timesteps=num_timesteps,
                                                            num_time_seg=num_time_seg, graph_param=graph_param,
                                                            sampling_param=sampling_param,
                                                            nflow_version=nflow_version)

        torch.save(test_data, test_data_path)
        print("Test data saved.")
    else:
        print("Test data already exists.")


def gridsearch(data_path, result_name, lam_log_space, mu_log_space, rank, inv_layers=None, num_iter=100,
               init="randsc", alg="bsca"):
    # file_name = "gridsearch_ref_r20_ON_r2_10iter"
    if inv_layers is None:
        inv_layers = [10, 100]

    data = utils.retrieve_data(data_path)
    data.to(DEVICE)

    print("## Gridsearch ## {}".format(result_name))
    result_path = os.path.join(RESULTDIR, result_name + DFILEEXT)

    if os.path.isfile(result_path):
        print("Result already exists.")
        return

    auc_results = torch.zeros(len(lam_log_space), len(mu_log_space), len(inv_layers), dtype=FP_DTYPE)

    for lam_idx in range(len(lam_log_space)):
        for mu_idx in range(len(mu_log_space)):
            print("Lam Idx {} | Mu Idx {}".format(lam_idx, mu_idx))
            lam = torch.exp(lam_log_space[lam_idx]).to(DEVICE)
            mu = torch.exp(mu_log_space[mu_idx]).to(DEVICE)

            torch.manual_seed(0)  # fixing initialization
            if alg == "bsca":
                A = reference_algo.bsca_incomplete_meas(data, lam, mu, rank, num_iter, init=init, return_im_steps=True)[
                    2]
            elif alg == "bbcd":
                A = reference_algo.batch_bcd_incomplete_meas(data, lam, mu, rank, num_iter, init=init,
                                                             return_im_steps=True)[2]
            elif alg == "bbcd_r":
                A = \
                    reference_algo.batch_bcd_incomplete_meas(data, lam, mu, rank, num_iter, init=init,
                                                             order="PQA", return_im_steps=True)[2]
            elif alg == "bsca_tens_nrlx":
                A = \
                    unrolled_bsca_tensor.bsca_tensor(data, lam, mu, None, rank, num_iter=num_iter, num_time_seg=None,
                                                     option="nrlx", nnf=True, return_im_steps=True)[4]
            elif alg == "bsca_tens_rlx":
                A = \
                    unrolled_bsca_tensor.bsca_tensor(data, lam, mu, None, rank, num_iter=num_iter, num_time_seg=None,
                                                     option="rlx", nnf=True, return_im_steps=True)[4]
            else:
                raise ValueError
            A = A[inv_layers]

            # stats_ref = evalfun.detector_single_class_auc_approx(test_data, A_ref, batch_mean=True)
            # auc_ref = stats_ref["auc"]
            auc_ref = evalfun.exact_auc(A.abs(), data["A"]).cpu()

            auc_results[lam_idx, mu_idx] = auc_ref

    results = {"auc": auc_results, "rank": rank, "lam_log": lam_log_space, "mu_log": mu_log_space,
               "result_name": result_name, "layers": inv_layers, "init": init}

    torch.save(results, result_path)
    print("Saved results")
    return


def show_results_gridsearch(result_name, layers_to_show=None):
    result_path = os.path.join(RESULTDIR, result_name + DFILEEXT)
    results = torch.load(result_path)
    auc = results["auc"]
    lam_log_space = results["lam_log"]
    mu_log_space = results["mu_log"]
    layers = results["layers"]
    rank = results["rank"]

    LAM, MU = torch.meshgrid(lam_log_space, mu_log_space)

    zmin = 0.5
    zmax = 1
    if layers_to_show is None:
        layers_to_show = layers
        layers_to_show_idx = list(range(len(layers)))
    else:
        assert (all([e in layers for e in layers_to_show]))
        layers_to_show_idx = [layers.index(e) for e in layers_to_show]

    ## Absolut maximum AUC
    auc = auc.numpy()
    max_idx = np.unravel_index(auc.argmax(), auc.shape)
    lam_log_max = lam_log_space[max_idx[0]]
    mu_log_max = mu_log_space[max_idx[1]]
    print("Absolute best achieved AUC \n Iteration {}: Best AUC: {}, log(lambda)={}, best log(mu)={}".format(
        layers[max_idx[2]],
        auc.max(),
        lam_log_max,
        mu_log_max))

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array([0] + layers), np.concatenate([np.array([0.5]), auc[max_idx[0], max_idx[1]]]))
    ax.vlines(layers[max_idx[2]], 0.5, 1.0, colors="r")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("AUC")
    ax.set_title("Max AUC run")
    ax.set_ylim([0.5, 1.0])

    fig, axs = plt.subplots(1, len(layers_to_show))
    for i in range(len(layers_to_show)):
        idx = layers_to_show_idx[i]
        auc_temp = auc[:, :, idx]
        if len(layers_to_show) > 1:
            ax_obj = axs[i]
        else:
            ax_obj = axs
        c = ax_obj.pcolormesh(LAM, MU, auc[:, :, idx], cmap="seismic", vmin=zmin, vmax=auc_temp.max())
        ax_obj.set_xlabel("log(lambda)")
        ax_obj.set_ylabel("log(mu)")

        ax_obj.set_title("{} Iterations, AUC_max={}".format(layers_to_show[i], auc_temp.max()))
        max_idx = np.unravel_index(auc_temp.argmax(), auc_temp.shape)
        ax_obj.scatter(lam_log_space[max_idx[0]], mu_log_space[max_idx[1]])
        print("Iteration {}: Best AUC: {}, log(lambda)={}, best log(mu)={}".format(layers_to_show[i],
                                                                                   auc_temp.max(),
                                                                                   lam_log_space[max_idx[0]],
                                                                                   mu_log_space[max_idx[1]]))

        export_path = os.path.join(EXPORTDIR, result_name + "_iter{}".format(layers_to_show[i]) + ".txt")
        np.savetxt(export_path, _export_for_heatmap(lam_log_space, mu_log_space, auc_temp), delimiter="\t")
    # fig.suptitle("Grid Search Reference Alg. R6 on R6 Data")
    fig.suptitle("Grid Search R{} {}".format(rank, result_name))
    # fig.suptitle("Grid Search Reference Alg. R20 on R2 Data")
    fig.tight_layout()
    fig.colorbar(c, ax=axs)
    plt.show()

    return


# def bsca_param_search(batch_size, scenario_param, rank, num_iter, lam0_mu0_log_numpy, alg="bsca"):
#     utils.set_rng_seed(0)
#     data = datagen.generate_synthetic_nw_scenarios(batch_size=batch_size, **scenario_param,
#                                                    nflow_version="exp+exp")
#     data.to(DEVICE)
#     lam0_mu0_log = lam0_mu0_log_numpy
#
#     def run_bsca(lam_mu_log):
#         lam = torch.tensor(lam_mu_log[0]).exp().type(FP_DTYPE)
#         mu = torch.tensor(lam_mu_log[1]).exp().type(FP_DTYPE)
#         torch.manual_seed(0)  # fixing initialization
#         if alg == "bsca":
#             A = \
#                 reference_algo.bsca_incomplete_meas(data, lam, mu, rank, num_iter, init="randsc",
#                                                     return_im_steps=False)[2]
#         elif alg == "bsca_tens":
#             A = unrolled_bsca_tensor.bsca_tensor(data, lam, mu, None, rank, num_iter=num_iter,
#                                                  num_time_seg=None, option="nrlx", nnf=True, return_im_steps=False)[4]
#         else:
#             raise ValueError
#         # A_ref = A_ref[-1]
#         auc = evalfun.exact_auc(A.abs(), data["A"]).cpu()
#         print("Sample", auc, lam, mu)
#         return -auc.numpy()
#
#     res = scipy.optimize.minimize(fun=run_bsca, x0=lam0_mu0_log, method="Nelder-Mead", options={"xatol": 0.01})
#     assert (res.success)
#     lam_mu_log_opt = res.x
#     obj = -np.array([res.fun])
#
#     return lam_mu_log_opt, obj


def get_run_name(data_name, nn_model_class, num_layers, rank, **kwargs):
    run_name = "{}_{}_ly{}_r{}".format(nn_model_class.__name__, data_name, num_layers, rank)
    if "shared_weights" in kwargs and kwargs["shared_weights"]:
        run_name = run_name + "_sw"

    if "no_tensor" in kwargs and kwargs["no_tensor"]:
        run_name = run_name + "_notens"
    if "param_nw" in kwargs:
        if kwargs["param_nw"] == "bayes":
            run_name = run_name + "_bay"
        elif kwargs["param_nw"] == "modewise":
            run_name = run_name + "_mw_"
        elif kwargs["param_nw"] == "modewise_sw":
            run_name = run_name + "_mwsw_"
        elif kwargs["param_nw"]:
            run_name = run_name + "_paramnw"
    if "datafit_options" in kwargs and kwargs["datafit_options"]:
        if "static1" in kwargs["datafit_options"]:
            run_name = run_name + "df1"
        elif "static2" in kwargs["datafit_options"]:
            run_name = run_name + "df2"
        elif "static2e-3" in kwargs["datafit_options"]:
            run_name = run_name + "df2e-3"
        elif "static2e-0" in kwargs["datafit_options"]:
            run_name = run_name + "df2e-0"
        elif "static2clamp" in kwargs["datafit_options"]:
            run_name = run_name + "df2c"
        elif "staticn1ly" in kwargs["datafit_options"]:
            run_name = run_name + "dfn1ly"
        elif "staticn2ly" in kwargs["datafit_options"]:
            run_name = run_name + "dfn2ly"
        elif "staticnc1ly" in kwargs["datafit_options"]:
            run_name = run_name + "dfnc1ly"
        elif "staticnc2ly" in kwargs["datafit_options"]:
            run_name = run_name + "dfnc2ly"
        elif "dynnc1ly" in kwargs["datafit_options"]:
            run_name = run_name + "dfdyn1ly"
        elif "var" in kwargs["datafit_options"]:
            run_name = run_name + "dfv"
        elif "1dyn1ly" in kwargs["datafit_options"]:
            run_name = run_name + "df1ly"
        elif "f1ly" in kwargs["datafit_options"]:
            run_name = run_name + "dff1ly"
        elif "f2ly" in kwargs["datafit_options"]:
            run_name = run_name + "dff2ly"

    if "lam_options" in kwargs and kwargs["lam_options"]:
        if "1dyn1ly" in kwargs["lam_options"]:
            run_name = run_name + "lm1"

    if "mu_options" in kwargs and kwargs["mu_options"]:
        if "static1" in kwargs["mu_options"]:
            run_name = run_name + "mu1"
        elif "dyn1ly" in kwargs["mu_options"]:
            run_name = run_name + "mud1ly"
        elif "dynnc1ly" in kwargs["mu_options"]:
            run_name = run_name + "mudc1ly"
        elif "dy2nc1ly" in kwargs["mu_options"]:
            run_name = run_name + "mud2c1ly"
        elif "dy2nc1lyi" in kwargs["mu_options"]:
            run_name = run_name + "mud2c1lyi"
        elif "dy3nc1ly" in kwargs["mu_options"]:
            run_name = run_name + "mud3c1ly"
        elif "dy4nc1ly" in kwargs["mu_options"]:
            run_name = run_name + "mud4c1ly"
        elif "1dyn1ly" in kwargs["mu_options"]:
            run_name = run_name + "mu1ly"
        elif "1dyn1lyi" in kwargs["mu_options"]:
            run_name = run_name + "mu1lyi"
        elif "d1ly" in kwargs["mu_options"]:
            run_name = run_name + "mud1ly"
        elif "f1ly" in kwargs["mu_options"]:
            run_name = run_name + "muf1ly"
        elif "f2ly" in kwargs["mu_options"]:
            run_name = run_name + "muf2ly"

        if "feat_after_PQ" in kwargs["mu_options"]:
            run_name = run_name + "_fq"

    if "skip_connections" in kwargs and kwargs["skip_connections"]:
        run_name = run_name + "_skip"
    if "option" in kwargs:
        run_name = run_name + "_" + kwargs["option"]
    if "it1_option" in kwargs:
        run_name = run_name + "_it1" + kwargs["it1_option"]
    # if it1_option:
    #     run_name = run_name + "_it1" + it1_option
    if "balanced" in kwargs and kwargs["balanced"]:
        run_name = run_name + "_bld"
    if "prox" in kwargs and kwargs["prox"]:
        run_name = run_name + "_px"
    if "normalize" in kwargs and kwargs["normalize"]:
        run_name = run_name + "_nr"
    if "skipA" in kwargs and kwargs["skipA"] > 0:
        run_name = run_name + "_sa" + str(kwargs["skipA"])
    if "batch_norm" in kwargs and kwargs["batch_norm"]:
        run_name = run_name + "_bn"
    if "two_nu" in kwargs and kwargs["two_nu"]:
        run_name = run_name + "_tn"

    # hackery
    # if "opt_kw" in kwargs and "betas" in kwargs["opt_kw"]:
    #     run_name = run_name + "_bc"
    if "opt_kw" in kwargs and "algo" in kwargs["opt_kw"]:
        if kwargs["opt_kw"]["algo"] == "SGD":
            run_name = run_name + "_sgd"
    if "loss_type" in kwargs:
        if kwargs["loss_type"] == "approxauc2_homotopy":
            pass
        elif kwargs["loss_type"] == "logistic":
            run_name = run_name + "_llg"
        elif kwargs["loss_type"] == "approxauc2_fixed":
            run_name = run_name + "_aucfx"
        elif "approxauc_homotopy_ss" in kwargs["loss_type"]:
            run_name = run_name + "_aucfs"
        elif "approxauc2_homotopy_ss" in kwargs["loss_type"]:
            run_name = run_name + "_aucfs"
        elif "approxauc2_fixed_ss" in kwargs["loss_type"]:
            run_name = run_name + "_aucfxfs"
        else:
            raise ValueError

    if "acronym" in kwargs and kwargs["acronym"]:
        run_name = run_name + "_" + kwargs["acronym"]

    return run_name


# def training_parametrized(result_dir, data_name, nn_model_class, num_layers, rank, batch_size=2,
#                           ovl_training_data_name=None, fixed_steps_per_epoch=None,
#                           skip_connections=False, param_nw=False, shared_weights=False,
#                           option=None, it1_option=None, balanced=False):

MODEL_KW = ["init", "param_nw", "shared_weights", "it1_option", "balanced", "prox", "normalize", "skipA", "no_tensor"]
LAYER_KW = ["skip_connections", "option", "datafit_options", "mu_options", "lam_options", "batch_norm", "two_nu"]


def do_training(run_path, train_data, valid_data,
                nn_model_class, num_layers, rank,
                num_epochs=500, batch_size=10, batch_partition_size=None,
                opt_kw=None, sched_kw=None, weight_decay=None,
                loss_type="approxauc2_homotopy", loss_options=None, force_num_steps_per_epoch=None,
                **model_param):
    """

    :param run_path:
    :param train_data:
    :param valid_data:
    :param nn_model_class:
    :param num_layers:
    :param rank:
    :param num_epochs:
    :param batch_size:
    :param batch_partition_size:
    :param opt_kw:
    :param sched_kw:
    :param weight_decay:
    :param loss_type:
    :param loss_options:
    :param force_num_steps_per_epoch:
    :param model_param:
    :return:
    """

    if sched_kw is None:
        sched_kw = {"milestones": [100, 300, 400], "gamma": 0.33}
    if opt_kw is None:
        opt_kw = {"lr": 0.01, "weight_decay": 0.0}

    if os.path.isfile(run_path):
        print("Run {} already exists.".format(run_path))
        return

    nn_model_dict = {"num_layers": num_layers, "rank": rank, "init": "randsc", "layer_param": {}}
    for k, e in model_param.items():
        if k in MODEL_KW:
            nn_model_dict[k] = e
        elif k in LAYER_KW:
            nn_model_dict["layer_param"][k] = e
        elif k in nn_model_dict.keys():
            pass  # check whether it has purpose
        else:
            raise ValueError

    report = training.training_general(run_path, train_data, valid_data,
                                       nn_model_class, nn_model_dict,
                                       num_epochs, batch_size, opt_kw, sched_kw,
                                       batch_partition_size=batch_partition_size,
                                       loss_type=loss_type, loss_options=loss_options, weight_decay=weight_decay,
                                       fixed_steps_per_epoch=force_num_steps_per_epoch)

    torch.save(report, run_path)
    print("Run {} saved.".format(run_path))


def eval_model(run_path, nn_model_class, test_data, auc_over_layers=True, training_stats=False, computetime=False, roc=False):
    # run_name = training_data_name + "_ly{}_r{}".format(num_layers, rank)
    report = torch.load(run_path, map_location=torch.device("cpu"))

    run_name = os.path.splitext(os.path.basename(run_path))[0]

    model_dict = report["model_dict"]
    model_kw = report["model_kw"]

    # model_class = report["model_class"]
    model_class = nn_model_class
    model = model_class(**model_kw)
    model.load_state_dict(model_dict)
    model.to(DEVICE)
    model.eval()  # IMPORTANT
    num_layers = model.num_layers

    if auc_over_layers or computetime:
        if isinstance(test_data, dict):
            test_data_path = test_data["path"]
            test_data_name = os.path.splitext(os.path.basename(test_data_path))[0]
            test_data_set = torch.load(test_data_path)
            test_data_set = datagen.nw_scenario_subset(test_data_set, test_data["idx"])
        else:
            test_data_path = os.path.join(SCENARIODIR, test_data + DFILEEXT)
            test_data_name = test_data
            test_data_set = torch.load(test_data_path)
        test_data_set.to(DEVICE)

        export_path_computetime = os.path.join(EXPORTDIR, "{}_ON_{}_time.txt".format(run_name, test_data_name))
        if not os.path.isfile(export_path_computetime) and computetime:
            with torch.autograd.no_grad():
                torch.manual_seed(0)
                s = time.time()
                dummy = model(test_data_set)[0]
                e = time.time()

            elapsed_time = e - s
            header = "num_layer\tnum_scenarios\tprocess_time"
            export_data_computetime = np.stack([num_layers, test_data_set["batch_size"], elapsed_time])[np.newaxis, :]
            np.savetxt(export_path_computetime, export_data_computetime, delimiter="\t", header=header)
        else:
            print("{} already exists".format(export_path_computetime))

        export_path_auclayers = os.path.join(EXPORTDIR, "{}_ON_{}_auc.txt".format(run_name, test_data_name))
        if not os.path.isfile(export_path_auclayers) and auc_over_layers:
            with torch.autograd.no_grad():
                torch.manual_seed(0)
                s = time.time()
                model_out = model(test_data_set)
                e = time.time()
            elapsed = e - s

            model_out_A = model_out[-1]
            auc_model = utils.exact_auc(model_out_A.abs(), test_data_set["A"]).cpu()
            mu = model.mu_val.cpu().numpy()

            # if lam.ndim == 2:
            #     lam = np.log(lam.mean(axis=-1))
            # else:
            #     lam = np.log(lam)
            if mu.ndim == 2:
                mu = np.log(mu).mean(axis=-1)
            else:
                mu = np.log(mu)

            mu = np.concatenate([np.array([-999.0]), mu])

            stk = [np.arange(num_layers + 1), auc_model.numpy(), mu]
            header = "Layer\tAUC\tmu"

            if hasattr(model, "lam_val"):
                lam = model.lam_val.cpu().numpy()
                if lam.ndim == 2:
                    lam = np.log(lam).mean(axis=-1)
                else:
                    lam = np.log(lam)
                lam = np.concatenate([np.array([-999.0]), lam])
                stk.append(lam)
                header = header + "\tlam"

            if hasattr(model, "nu_val"):
                nu = model.nu_val.cpu().numpy()
                if nu.ndim == 2:
                    nu = np.log(nu).mean(axis=-1)
                else:
                    nu = np.log(nu)
                nu = np.concatenate([np.array([-999.0]), nu])
                stk.append(nu)
                header = header + "\tnu"

            export_data_auc = np.stack(stk, axis=-1)
            header = header + "\tElapsed_time={}".format(elapsed)

            np.savetxt(export_path_auclayers, export_data_auc, delimiter="\t", header=header)
        elif auc_over_layers:
            print("{} already exists".format(export_path_auclayers))

        # export_path_roc = os.path.join(EXPORTDIR, "{}_ON_{}_roc.txt".format(run_name, test_data_name))
        export_path_roc_full = os.path.join(EXPORTDIR, "{}_ON_{}_roc.pt".format(run_name, test_data_name))
        if not os.path.isfile(export_path_roc_full) and roc:
            with torch.autograd.no_grad():
                torch.manual_seed(0)
                s = time.time()
                model_out = model(test_data_set)
                e = time.time()
            model_out_A = model_out[-1][-1]  # last layer

            # roc_curve = evalfun.roc_curve(model_out_A.abs().cpu(), test_data_set["A"].cpu(), num_samples=50)
            # header = "pfa_roc\tpd_roc"
            # export_data_roc = np.stack([roc_curve["pfa_roc"].numpy(), roc_curve["pd_roc"].numpy()], axis=-1)
            # np.savetxt(export_path_roc, export_data_roc, delimiter="\t", header=header)

            roc_curve = evalfun.roc_curve(model_out_A.abs().cpu(), test_data_set["A"].cpu(), num_samples=500, batch_mean=False)
            torch.save(roc_curve, export_path_roc_full)
        elif roc:
            print("{} already exists".format(export_path_roc_full))

    if training_stats:
        export_path_tsteps = os.path.join(EXPORTDIR, "{}_tloss.txt".format(run_name))
        if not os.path.isfile(export_path_tsteps):
            training_loss = report["training_loss"].numpy()
            export_tsteps = np.stack([np.arange(len(training_loss)), training_loss], axis=-1)
            np.savetxt(export_path_tsteps, export_tsteps, delimiter="\t", header="Layer\ttraining_data_loss")
        else:
            pass

        export_path_tepochs = os.path.join(EXPORTDIR, "{}_tepochs.txt".format(run_name))
        if not os.path.isfile(export_path_tepochs):
            val_loss = report["test_loss"].numpy()
            anomaly_l2 = report["test_anomaly_l2"].numpy()
            auc = report["test_det_eval"]["auc"].numpy()
            epochs = np.arange(len(val_loss))

            if "lam" in report["reg_param"] and len(report["reg_param"]["lam"]) > 0:
                lam = report["reg_param"]["lam"]
                if lam.ndim == 3:
                    lam_log = lam.log().mean(dim=-1)  # was previously the other way around
                else:
                    lam_log = lam.log()

            if "nu" in report["reg_param"] and len(report["reg_param"]["nu"]) > 0:
                nu = report["reg_param"]["nu"]
                if nu.ndim == 3:
                    nu_log = nu.log().mean(dim=-1)  # was previously the other way around
                else:
                    nu_log = nu.log()

            mu = report["reg_param"]["mu"]
            if mu.ndim == 3:
                mu_log = mu.log().mean(dim=-1)  # was previously the other way around
            else:
                mu_log = mu.log()

            export_data = [epochs, val_loss, auc, anomaly_l2[:, -1]]
            export_data_header = "epoch\tval_loss\tval_auc\tanomaly_l2"
            for l in range(num_layers):
                if "lam" in report["reg_param"] and len(report["reg_param"]["lam"]) > 0:
                    export_data.append(lam_log[..., l].cpu())
                    export_data_header = export_data_header + "\tlam{}".format(l + 1)
                if "nu" in report["reg_param"] and len(report["reg_param"]["nu"]) > 0:
                    export_data.append(nu_log[..., l].cpu())
                    export_data_header = export_data_header + "\tnu{}".format(l + 1)
                export_data.append(mu_log[..., l].cpu())
                export_data_header = export_data_header + "\tmu{}".format(l + 1)

            export_data = np.stack(export_data, axis=-1)

            np.savetxt(export_path_tepochs, export_data, delimiter="\t", header=export_data_header)
        else:
            pass


def _export_for_heatmap(x, y, z):
    size = len(x) * len(y)
    assert (size == z.size)
    filemat = []
    for i in range(len(x)):
        for j in range(len(y)):
            filemat.append(np.array([x[i], y[j], z[i, j]]))
    filemat = np.stack(filemat, axis=0)
    return filemat


from reference_algo_hankel import subspace_tracking_anomography
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


def _kasai_refalg_splitbatch(batch_data_dict, rank, lam, mu_r, mu_h, mu_s, win_len, partition_size=1):
    """Curbs memory demand of method by splitting the batch."""
    U_list = []
    V_list = []
    W_list = []
    A_list = []
    assert batch_data_dict["batch_size"] % partition_size == 0  # for sanity
    sidx_list = [list(range(s, s + partition_size)) for s in range(0, batch_data_dict["batch_size"], partition_size)]
    for sidx in sidx_list:
        sample = datagen.nw_scenario_subset(batch_data_dict, sidx)
        Us, Vs, Ws, As = subspace_tracking_anomography(sample, rank, lam, mu_r, mu_h, mu_s, win_len)
        U_list.append(Us)
        V_list.append(Vs)
        W_list.append(Ws)
        A_list.append(As)

    U = torch.cat(U_list, dim=0)
    V = torch.cat(V_list, dim=0)
    W = torch.cat(W_list, dim=0)
    A = torch.cat(A_list, dim=0)
    return U, V, W, A


def _bayes_opt_kasai_refalg(data_dict, rank, window_length=None, pinit=None, pscale=None, num_fun_calls=250,
                            batch_partition_size=1, search_space=None):
    UNCERTAINTY = 1e-4  # estimated variance
    # win_len = window_length
    # if pinit is None:
    #     pinit = [1.0, 1.0, 1.0, 1.0, 10.0]

    if window_length is not None:
        def convert_param(par):
            lam_cmp, mu_r_log, mu_h_cmp, mu_s_log = par
            win_len = window_length
            # lam_cmp, mu_r_log, mu_h_cmp, mu_s_log, win_len = par
            lam = torch.sigmoid(torch.tensor(lam_cmp))
            mu_r = torch.exp(torch.tensor(mu_r_log))
            mu_h = torch.sigmoid(torch.tensor(mu_h_cmp))  # this is due to the algorithm being unstable if mu_h > 1
            mu_s = torch.exp(torch.tensor(mu_s_log))

            return lam, mu_r, mu_h, mu_s, win_len

        n_initial = 16 * 4
        # def refalg_sta(params):
        #     # lam, mu_r, mu_h, mu_s, win_len = convert_param(params)
        #     lam, mu_r, mu_h, mu_s = convert_param(params)
        #     print("Calling method using parameters lam={:.3f}, mu_r={:.3f}, mu_h={:.3f}, mu_s={:.3f}, winlen={}".format(lam, mu_r, mu_h, mu_s, win_len))
        #     ### Splitting data because memory is out of control for high W
        #
        #     _, _, _, A = _kasai_refalg_splitbatch(data_dict, rank, lam, mu_r, mu_h, mu_s, win_len, split_size=batch_split_size)
        #     nauc = - evalfun.exact_auc(A, data_dict["A"])
        #     print("AUC={}".format(-nauc))
        #     return nauc.item()

    else:
        def convert_param(par):
            lam_cmp, mu_r_log, mu_h_cmp, mu_s_log, win_len = par
            lam = torch.sigmoid(torch.tensor(lam_cmp))
            mu_r = torch.exp(torch.tensor(mu_r_log))
            mu_h = torch.sigmoid(torch.tensor(mu_h_cmp))  # this is due to the algorithm being unstable if mu_h > 1
            mu_s = torch.exp(torch.tensor(mu_s_log))

            return lam, mu_r, mu_h, mu_s, win_len

        n_initial = 32 * 4

    def refalg_sta(params):
        lam, mu_r, mu_h, mu_s, win_len = convert_param(params)
        print("Calling method using parameters lam={:.3f}, mu_r={:.3f}, mu_h={:.3f}, mu_s={:.3f}, winlen={}".format(
            lam, mu_r, mu_h, mu_s, win_len))
        ### Splitting data because memory is out of control for high W

        _, _, _, A = _kasai_refalg_splitbatch(data_dict, rank, lam, mu_r, mu_h, mu_s, win_len,
                                              partition_size=batch_partition_size)
        nauc = - evalfun.exact_auc(A, data_dict["A"])
        print("AUC={}".format(-nauc))
        return nauc.item()

    if search_space is None:
        if window_length is not None:
            search_space = [(-5.0, 5.0),
                            (-8.0, 2.0),
                            (-8.0, 2.0),
                            (-8.0, 2.0)]
        else:
            search_space = [(-5.0, 5.0),
                            (-8.0, 2.0),
                            (-8.0, 2.0),
                            (-8.0, 2.0),
                            (2, 50)]

    if pscale is None:
        if window_length is not None:
            pscale = [1.0, 1.0, 1.0, 1.0]
        else:
            pscale = [1.0, 1.0, 1.0, 1.0, 1.0]

    print("Starting BayesOpt on reference algorithm.")

    ### Params lam, mu_r, mu_h, mu_s, window_length
    pscale = np.array(pscale)
    ker = Matern(nu=2.5, length_scale=pscale)  # + 0.5  # AUC is >= 0.5
    gpr = GaussianProcessRegressor(kernel=ker, alpha=UNCERTAINTY)

    # if search_space is None:
    #     search_space = [(-5.0, 5.0),
    #                     (-8.0, 2.0),
    #                     (-8.0, 2.0),
    #                     (-8.0, 2.0),
    #                     (2, 50)]

    res = gp_minimize(
        refalg_sta,
        search_space,
        base_estimator=gpr,
        acq_func="EI",
        n_calls=num_fun_calls,
        n_initial_points=n_initial,
        xi=0.005,  # minimum improvement
        initial_point_generator="lhs"  # latin hypercube sequence
    )
    print("Finished BayesOpt on reference algorithm.")

    p_opt = res.x
    auc_opt = -res.fun

    p_list = res.x_iters
    auc_list = [-r for r in res.func_vals]

    out_dict = {"p_opt": p_opt, "auc_opt": auc_opt, "p_list": p_list, "auc_list": auc_list}

    for p, auc in zip(p_list, auc_list):
        # lam, mu_r, mu_h, mu_s, win_len = convert_param(p)
        lam, mu_r, mu_h, mu_s, win_len = convert_param(p)
        print('Achieved {:.3f} with lam={:.3f}, mu_r={:.3f}, mu_h={:.3f}, mu_s={:.3f}, winlen={}'.format(auc, lam, mu_r,
                                                                                                         mu_h, mu_s,
                                                                                                         win_len))

    # lam, mu_r, mu_h, mu_s, win_len = convert_param(p_opt)
    lam, mu_r, mu_h, mu_s, win_len = convert_param(p_opt)
    opt_param = {"lam": lam, "mu_r": mu_r, "mu_h": mu_h, "mu_s": mu_s, "win_len": win_len}
    # opt_param = {"lam": lam, "mu_r": mu_r, "mu_h": mu_h, "mu_s": mu_s}

    return auc_opt, opt_param, out_dict


def eval_kasai_refalg(training_data_path, val_data_path, result_name, rank, window_length=None, batch_partition_size=1,
                      search_space=None, num_fun_calls=100, roc=False):
    training_data = utils.retrieve_data(training_data_path)
    val_data = utils.retrieve_data(val_data_path)

    # moving to device
    # print("Using device {}".format(DEVICE))
    training_data.to(DEVICE)
    val_data.to(DEVICE)

    opt_result_path = os.path.join(RESULTDIR, result_name + DFILEEXT)
    print("Running {}".format(result_name))
    if not os.path.isfile(opt_result_path):
        auc_train, opt_param_train, opt_dict = _bayes_opt_kasai_refalg(training_data,
                                                                       rank,
                                                                       window_length=window_length,
                                                                       batch_partition_size=batch_partition_size,
                                                                       search_space=search_space,
                                                                       num_fun_calls=num_fun_calls)
        torch.save({"opt_dict": opt_dict, "auc_train": auc_train, "opt_param_train": opt_param_train}, opt_result_path)
    else:
        print("Parameters of {} already found.".format(result_name))
        saved_opt = torch.load(opt_result_path)
        if not "opt_param_train" in saved_opt:  # only legacy
            lam_cmp, mu_r_log, mu_h_cmp, mu_s_log, win_len = saved_opt["p_opt"]
            lam = torch.sigmoid(torch.tensor(lam_cmp))
            mu_r = torch.exp(torch.tensor(mu_r_log))
            mu_h = torch.sigmoid(torch.tensor(mu_h_cmp))  # this is due to the algorithm being unstable if mu_h > 1
            mu_s = torch.exp(torch.tensor(mu_s_log))
            opt_param_train = {"lam": lam, "mu_r": mu_r, "mu_h": mu_h, "mu_s": mu_s, "win_len": win_len}
            auc_train = saved_opt["auc_opt"]
        else:
            auc_train, opt_param_train, opt_dict = saved_opt["auc_train"], saved_opt["opt_param_train"], saved_opt["opt_dict"]

    export_path = os.path.join(EXPORTDIR, result_name + ".txt")
    if not os.path.isfile(export_path):
        s = time.time()
        _, _, _, A = _kasai_refalg_splitbatch(val_data, rank, **opt_param_train, partition_size=batch_partition_size)
        e = time.time()
        # _, _, _, A = _kasai_refalg_splitbatch(val_data, rank, window_length, **opt_param_train)
        elapsed = e - s

        auc_val = evalfun.exact_auc(A, val_data["A"])
        auc_val = auc_val.cpu()

        export_header = "AUC_train\tAUC_val\tElapsed_time"
        export_data = [auc_train, auc_val, elapsed]
        for k, v in opt_param_train.items():
            export_header = export_header + "\t" + k
            export_data.append(v)
        export_data = np.stack(export_data)[None, :]
        np.savetxt(export_path, export_data, delimiter="\t", header=export_header)
        print("Result saved in {}".format(export_path))
    else:
        print("{} already exists.".format(export_path))

    # export_path_roc = os.path.join(EXPORTDIR, result_name + "_roc.txt")
    export_path_roc_full = os.path.join(EXPORTDIR, result_name + "_roc.pt")
    if not os.path.isfile(export_path_roc_full) and roc:
        _, _, _, A = _kasai_refalg_splitbatch(val_data, rank, **opt_param_train)
        # roc_curve = evalfun.roc_curve(A.abs().cpu(), val_data["A"].cpu(), num_samples=250)
        # header = "pfa_roc\tpd_roc"
        # export_data_roc = np.stack([roc_curve["pfa_roc"].numpy(), roc_curve["pd_roc"].numpy()], axis=-1)
        # np.savetxt(export_path_roc, export_data_roc, delimiter="\t", header=header)

        roc_curve = evalfun.roc_curve(A.abs().cpu(), val_data["A"].cpu(), num_samples=500,
                                      batch_mean=False)
        torch.save(roc_curve, export_path_roc_full)
    elif roc:
        print("{} already exists".format(export_path_roc_full))


def _classical_alg(data_dict, rank, num_iter=None, lam=None, mu=None, nu=None, init="randsc", alg="bsca", return_im_steps=False,
                   return_obj_val=False):
    # print(rank, num_iter, lam, mu, nu, init, alg)
    s = time.time()
    if alg == "bsca":
        P, Q, A = \
            reference_algo.bsca_incomplete_meas(data_dict, lam, mu, rank, num_iter, init=init,
                                                return_im_steps=return_im_steps)
    elif alg == "bbcd":
        P, Q, A = reference_algo.batch_bcd_incomplete_meas(data_dict, lam, mu, rank, num_iter, init=init,
                                                           return_im_steps=return_im_steps)
    elif alg == "bbcd_r":
        P, Q, A = \
            reference_algo.batch_bcd_incomplete_meas(data_dict, lam, mu, rank, num_iter, init=init,
                                                     order="PQA", return_im_steps=return_im_steps)
    elif alg == "bsca_tens_nrlx":
        X, U, V, W, A = \
            unrolled_bsca_tensor.bsca_tensor(data_dict, lam, mu, None, rank, num_iter=num_iter, num_time_seg=None,
                                             option="nrlx", nnf=True, return_im_steps=return_im_steps)
    elif alg == "bsca_tens_rlx":
        X, U, V, W, A = \
            unrolled_bsca_tensor.bsca_tensor(data_dict, lam, mu, nu, rank, num_iter=num_iter, num_time_seg=None,
                                             option="rlx", nnf=True, return_im_steps=return_im_steps)
    elif alg == "bsca_tens_rlx_it1nrlx":
        X, U, V, W, A = \
            unrolled_bsca_tensor.bsca_tensor(data_dict, lam, mu, nu, rank, num_iter=num_iter, num_time_seg=None,
                                             option="rlx", it1_option="nrlx", nnf=True, return_im_steps=return_im_steps)
    else:
        raise ValueError
    e = time.time()
    elapsed = e - s

    if return_obj_val:
        if alg in ["bsca", "bbcd", "bbcd_r"]:
            obj = reference_algo.network_anomalography_obj(data_dict, P, Q, A, lam, mu, batch_mean=True)
        elif alg == "bsca_tens_nrlx":
            obj = unrolled_bsca_tensor.obj_cpd_relaxed(data_dict, X, U, V, W, A, lam, mu, 0.0, batch_mean=True)
        elif alg == "bsca_tens_rlx" or alg == "bsca_tens_rlx_it1nrlx":
            obj = unrolled_bsca_tensor.obj_cpd_relaxed(data_dict, X, U, V, W, A, lam, mu, nu, batch_mean=True)
        else:
            raise ValueError

        return A, elapsed, obj

    else:
        return A, elapsed


def _bayes_classical_alg(data_dict, rank, num_fun_calls=250, alg="bsca", init="randsc"):
    UNCERTAINTY = 1e-4  # estimated variance

    # win_len = window_length
    # if pinit is None:
    #     pinit = [1.0, 1.0, 1.0, 1.0, 10.0]

    if alg in ["bsca", "bbcd", "bbcd_r", "bsca_tens_nrlx"]:
        def convert_param(par):
            num_iter, lam_log, mu_log = par
            lam = torch.exp(torch.tensor(lam_log, device=DEVICE))
            mu = torch.exp(torch.tensor(mu_log, device=DEVICE))
            return num_iter, lam, mu, 0.0

        search_space = [(1, 100),
                        (-10.0, 4.0),
                        (-10.0, 4.0), ]
        pscale = [5.0, 1.0, 1.0]
        n_initial = 8*4

    elif alg in ["bsca_tens_rlx", "bsca_tens_rlx_it1nrlx"]:
        def convert_param(par):
            num_iter, lam_log, mu_log, nu_log = par
            lam = torch.exp(torch.tensor(lam_log, device=DEVICE))
            mu = torch.exp(torch.tensor(mu_log, device=DEVICE))
            nu = torch.exp(torch.tensor(nu_log, device=DEVICE))
            return num_iter, lam, mu, nu

        search_space = [(1, 100),
                        (-10.0, 4.0),
                        (-10.0, 4.0),
                        (-10.0, 8.0), ]
        pscale = [5.0, 1.0, 1.0, 1.0]
        n_initial = 16*4
    else:
        raise ValueError

    def refalg_sta(params):
        num_iter, lam, mu, nu = convert_param(params)
        print(
            "Calling method using parameters num_layers={:.3f}, lam={:.3f}, mu={:.3f}, nu={:.3f}".format(num_iter, lam,
                                                                                                         mu, nu))

        A, _ = _classical_alg(data_dict, rank, num_iter, lam, mu, nu, alg=alg, return_im_steps=False)

        nauc = -evalfun.exact_auc(A, data_dict["A"])
        print("AUC={}".format(-nauc))
        return nauc.cpu().item()

    print("Starting BayesOpt on algorithm.")

    ### Params lam, mu_r, mu_h, mu_s, window_length
    pscale = np.array(pscale)
    ker = Matern(nu=2.5, length_scale=pscale)  # + 0.5  # AUC is >= 0.5
    gpr = GaussianProcessRegressor(kernel=ker, alpha=UNCERTAINTY)

    # if search_space is None:
    #     search_space = [(-5.0, 5.0),
    #                     (-8.0, 2.0),
    #                     (-8.0, 2.0),
    #                     (-8.0, 2.0),
    #                     (2, 50)]

    res = gp_minimize(
        refalg_sta,
        search_space,
        base_estimator=gpr,
        acq_func="EI",
        n_calls=num_fun_calls,
        n_initial_points=n_initial,
        xi=0.005,  # minimum improvement
        initial_point_generator="lhs"  # latin hypercube sequence
    )
    print("Finished BayesOpt on reference algorithm.")

    p_opt = res.x
    auc_opt = -res.fun

    p_list = res.x_iters
    auc_list = [-r for r in res.func_vals]

    out_dict = {"p_opt": p_opt, "auc_opt": auc_opt, "p_list": p_list, "auc_list": auc_list}

    for p, auc in zip(p_list, auc_list):
        num_iter, lam, mu, nu = convert_param(p)
        print("Achieved {:.3f} with num_iter={:.3f}, lam={:.3f}, mu={:.3f}, nu={:.3f}".format(auc, num_iter, lam, mu,
                                                                                              nu))

    num_iter, lam, mu, nu = convert_param(p_opt)
    opt_param = {"num_iter": num_iter, "lam": lam, "mu": mu, "nu": nu}

    return auc_opt, opt_param, out_dict


def eval_classical_alg(training_data_path, val_data_path, result_name, rank, num_fun_calls=100, alg="bsca", auc_over_iter=True, roc=False):
    training_data = utils.retrieve_data(training_data_path)
    val_data = utils.retrieve_data(val_data_path)

    # moving to device
    # print("Using device {}".format(DEVICE))
    training_data.to(DEVICE)
    val_data.to(DEVICE)

    opt_result_path = os.path.join(RESULTDIR, result_name + DFILEEXT)
    export_path1 = os.path.join(EXPORTDIR, result_name + ".txt")
    export_path2 = os.path.join(EXPORTDIR, result_name + "_iter" + ".txt")
    print("Running {}".format(result_name))

    if not os.path.isfile(opt_result_path):
        auc_train, opt_param_train, opt_dict = _bayes_classical_alg(training_data,
                                                                    rank,
                                                                    num_fun_calls=num_fun_calls,
                                                                    alg=alg)

        torch.save({"opt_dict": opt_dict, "auc_train": auc_train, "opt_param_train": opt_param_train}, opt_result_path)
    else:
        print("Parameters of {} already found.".format(result_name))
        saved_opt = torch.load(opt_result_path, map_location=DEVICE)
        auc_train, opt_param_train, opt_dict = saved_opt["auc_train"], saved_opt["opt_param_train"], saved_opt["opt_dict"]

    for k in opt_param_train.keys():
        if isinstance(opt_param_train[k], torch.Tensor):
            opt_param_train[k] = opt_param_train[k].to(DEVICE)

    if not os.path.isfile(export_path1):
        # print(opt_param_train, auc_train)
        # for p, auc in zip(opt_dict["p_list"], opt_dict["auc_list"]):
        #     num_iter, lam, mu = p
        #     print(
        #         "Achieved {:.3f} with num_iter={:.3f}, lam={:.3f}, mu={:.3f}".format(auc, num_iter, lam, mu))
        
        A, elapsed = _classical_alg(val_data, rank, **opt_param_train, return_im_steps=False, alg=alg)
        auc_val = evalfun.exact_auc(A, val_data["A"])
        auc_val = auc_val.cpu()

        export_header = "AUC_train\tAUC_val\tElapsed_time"
        export_data = [auc_train, auc_val, elapsed]
        for k, v in opt_param_train.items():
            export_header = export_header + "\t" + k
            export_data.append(v.cpu() if isinstance(v, torch.Tensor) else v)
        export_data = np.stack(export_data)[None, :]

        np.savetxt(export_path1, export_data, delimiter="\t", header=export_header)
        print("Result saved in {}".format(export_path1))
    else:
        print("{} already exists.".format(export_path1))

    if auc_over_iter:
        if not os.path.isfile(export_path2):
            """AUC and objective over iterations"""
            opt_param_train["num_iter"] = 1000
            A, _, obj_vals = _classical_alg(val_data, rank, **opt_param_train, return_im_steps=True, return_obj_val=True, alg=alg)
            obj_vals = obj_vals.cpu()
            auc_vals = evalfun.exact_auc(A, val_data["A"])
            auc_vals = auc_vals.cpu()

            num_iter = opt_param_train["num_iter"]
            itidx = torch.arange(num_iter + 1)
            result = torch.stack([itidx, obj_vals, auc_vals], dim=-1).numpy()
            np.savetxt(export_path2, result,
                       header="it\tobj\tauc  ### lam: {}, mu: {}, nu: {}".format(opt_param_train["lam"],
                                                                                 opt_param_train["mu"],
                                                                                 opt_param_train["nu"]),
                       delimiter="\t")

            print("Result saved in {}".format(export_path2))
        else:
            print("{} already exists.".format(export_path2))

    # export_path_roc = os.path.join(EXPORTDIR, result_name + "_roc.txt")
    export_path_roc_full = os.path.join(EXPORTDIR, result_name + "_roc.pt")
    if not os.path.isfile(export_path_roc_full) and roc:
        A, _ = _classical_alg(val_data, rank, **opt_param_train, return_im_steps=False, return_obj_val=False, alg=alg)
        # roc_curve = evalfun.roc_curve(A.abs().cpu(), val_data["A"].cpu(), num_samples=50)
        # header = "pfa_roc\tpd_roc"
        # export_data_roc = np.stack([roc_curve["pfa_roc"].numpy(), roc_curve["pd_roc"].numpy()], axis=-1)
        # np.savetxt(export_path_roc, export_data_roc, delimiter="\t", header=header)

        roc_curve = evalfun.roc_curve(A.abs().cpu(), val_data["A"].cpu(), num_samples=500,
                                      batch_mean=False)
        torch.save(roc_curve, export_path_roc_full)
    elif roc:
        print("{} already exists".format(export_path_roc_full))