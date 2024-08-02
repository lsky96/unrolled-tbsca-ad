import copy
import time
import os.path
from collections import defaultdict

import torch
import torch.optim as optim
import datagen
import evalfun
import lossfun
import utils

from config import DFILEEXT, DEVICE

DATADIR = os.path.abspath("scenario_data")
ROOTDIR = os.path.abspath("results")

LOAD_BALANCING = False


def training_general(run_name, training_data_set, validation_data_set, nn_model_class, nn_model_kw, num_epochs,
                     batch_size, opt_kw, sched_kw, batch_partition_size=None, loss_type="unsuper", loss_options=None, weight_decay=None,
                     fixed_steps_per_epoch=None):
    # report_path = os.path.join(ROOTDIR, run_path + DFILEEXT)
    """

    :param run_name: str
    :param training_data_set: path (without .pt) or dict {"path": PATH, "idx": LISTOFIDX}
    :param validation_data_set: path (without .pt) or dict {"path": PATH, "idx": LISTOFIDX}
    :param nn_model_class: str, model architecture
    :param nn_model_kw: dict, model architecture kwargs
    :param num_epochs: size of training_data_set // batch_size if not configured by fixed_steps_per_epoch
    :param batch_size: int, minibatch size
    :param opt_kw: dict, optimizer kwargs
    :param sched_kw: dict, scheduler kwargs
    :param batch_partition_size: keys ["train", val"], scenario batch is partitioned computation for memory optimization
    :param loss_type: type of loss function ()
    :param loss_options: dict, loss function kw
    :param weight_decay: dict, {i_epoch0: weight_decay_factor0, i_epoch1: weight_decay_factor1, ...}
    :param fixed_steps_per_epoch: workaround fixing number of steps per "epoch" regardless of dataset size
    :return:
    """

    print("Starting run {}".format(run_name))
    report = {"run_name": run_name,
              "model_class": nn_model_class.__name__,
              "model_kw": nn_model_kw,
              "training_data_set": training_data_set,
              "test_data_set": validation_data_set,  # old
              "validation_data_set": validation_data_set,
              "num_epochs": num_epochs,
              "fixed_steps_per_epoch": fixed_steps_per_epoch,  # DO NOT USE
              "batch_size": batch_size,
              "batch_partition_size": batch_partition_size,
              "opt_kw": opt_kw,
              "sched_kw": sched_kw,
              "loss_type": loss_type,
              "loss_options": loss_options,
              "weight_decay": weight_decay,
              "running": {"epoch": 0, "model_sd": None, "opt_sd": None, "sched_sd": None}
              }

    epoch = 0
    step = 0

    print("Loading data")
    if isinstance(training_data_set, dict):
        training_data = torch.load(training_data_set["path"])
        if "idx" in training_data_set.keys():
            training_data = datagen.nw_scenario_subset(training_data, training_data_set["idx"])
    else:
        training_data = torch.load(training_data_set + DFILEEXT)  # OLD
    training_data.to(DEVICE)

    if isinstance(validation_data_set, dict):
        val_data = torch.load(validation_data_set["path"])
        if "idx" in validation_data_set.keys():
            val_data = datagen.nw_scenario_subset(val_data, validation_data_set["idx"])
    else:
        val_data = torch.load(validation_data_set + DFILEEXT)  # OLD
    val_data.to(DEVICE)

    print("Initializing model")
    nn_model = nn_model_class(**nn_model_kw)  # creates model
    nn_model.to(DEVICE)

    if "algo" in opt_kw.keys():
        if opt_kw["algo"] == "SGD":
            print("SGD")
            opt_kw_tr = copy.deepcopy(opt_kw)
            del opt_kw_tr["algo"]
            optimizer = optim.SGD(nn_model.parameters(), **opt_kw_tr)
        else:
            raise ValueError
    else:
        optimizer = optim.AdamW(nn_model.parameters(), **opt_kw)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sched_kw)
    scheduler = MultiStepLRwWarmup(optimizer, **sched_kw)

    num_train_samples = training_data["batch_size"]
    num_val_samples = val_data["batch_size"]

    if batch_partition_size is not None:
        assert batch_size % batch_partition_size["train"] == 0
        assert num_val_samples % batch_partition_size["val"] == 0

    if fixed_steps_per_epoch:
        num_mbatches_epoch = fixed_steps_per_epoch
    else:
        num_mbatches_epoch = num_train_samples // batch_size  # number of mini batches per epoch

    # for t in range (max_sgd_steps):

    training_loss = []
    test_loss = []
    test_anomaly_l2 = []
    detector_eval = []

    reg_param = defaultdict(list)
    grad_tracking = {"max": [], "mean": []}
    grad_tracking_ly = {"max": [], "mean": []}
    # reg_param_ll = []

    # min_tstep_time = 999  # initialize with big value

    while epoch <= num_epochs:  # in epoch=0, only evaluation is done
        if epoch % 10 == 0:
            print("### Currently running: {} ###".format(run_name))

        print("### Epoch {} ###".format(epoch))

        # decide loss
        # loss_type_current = get_loss_option_current_epoch(loss_type, epoch)
        loss_type_current = loss_type

        if weight_decay is not None:
            if epoch in weight_decay.keys():
                for g in optimizer.param_groups:
                    g["weight_decay"] = weight_decay[epoch]
                print("### Changed weight decay to {} ###".format(weight_decay[epoch]))

        if epoch != 0:
            nn_model.train()
            # Shuffle the training data and labels
            permuted_indices = torch.randperm(num_train_samples)

            # Loop over mini batches
            for i_batch in range(num_mbatches_epoch):
                print("Step {}".format(i_batch + 1), end="")
                # Get the mini batch data and labels
                if fixed_steps_per_epoch and fixed_steps_per_epoch > num_train_samples // batch_size:
                    perm_idx_batch_idx = list(crange(i_batch * batch_size, (i_batch + 1) * batch_size, num_train_samples))
                    mbatch_indices = permuted_indices[perm_idx_batch_idx]
                else:
                    batch_start = i_batch * batch_size
                    batch_end = (i_batch + 1) * batch_size
                    mbatch_indices = permuted_indices[batch_start:batch_end]
                mbatch_data = datagen.nw_scenario_subset(training_data, mbatch_indices)

                optimizer.zero_grad(set_to_none=True)

                # with torch.autograd.detect_anomaly():
                if True:
                    # tstep_time = time.process_time()
                    if batch_partition_size is not None:
                        sidx_list = [list(range(s, s + batch_partition_size["train"])) for s in
                                     range(0, batch_size, batch_partition_size["train"])]
                        loss = []
                        for sidx in sidx_list:
                            mbatch_data_partition = datagen.nw_scenario_subset(mbatch_data, sidx)
                            model_out_partition = nn_model(mbatch_data_partition)
                            model_out_A_partition = model_out_partition[-1]
                            loss_partition = lossfun.lossfun_simple(mbatch_data_partition, model_out_A_partition, epoch,
                                                                    loss_type=loss_type_current, loss_options=loss_options)
                            loss_partition = loss_partition / len(sidx_list)
                            loss_partition.backward()
                            loss.append(loss_partition.detach())
                        loss = torch.stack(loss).sum()

                    else:
                        model_out = nn_model(mbatch_data)
                        model_out_A = model_out[-1]
                        loss = lossfun.lossfun_simple(mbatch_data, model_out_A, epoch, loss_type=loss_type_current,
                                                      loss_options=loss_options)
                        loss.backward()

                    """Reporting"""
                    training_loss.append(loss.detach().cpu())

                    all_param_grad = torch.cat([param.grad.view(-1) for param in nn_model.parameters() if param.grad is not None])
                    grad_tracking["max"].append(all_param_grad.abs().max().cpu())
                    grad_tracking["mean"].append(all_param_grad.abs().mean().cpu())

                    all_param_grad_ly = []
                    for ly in range(nn_model.num_layers):
                        all_param_grad_ly.append(torch.cat([param.grad.view(-1) for param
                                                            in nn_model.layers[ly].parameters()
                                                            if param.grad is not None]).abs())
                    # all_param_grad_ly = torch.stack(all_param_grad_ly)
                    grad_tracking_ly["max"].append(torch.stack([all_param_grad_ly[ly].max().cpu()
                                                             for ly in range(nn_model.num_layers)]))
                    grad_tracking_ly["mean"].append(torch.stack([all_param_grad_ly[ly].mean().cpu()
                                                             for ly in range(nn_model.num_layers)]))

                print(" - Training Loss={}".format(loss.detach().cpu()))
                # for n,p in nn_model.named_parameters():
                #     print(n,p)
                # for n, p in nn_model.named_parameters():
                #     print(n, p.grad)
                optimizer.step()  # performs one optimizer step, i.e., a gradient step on managed parameters
                step = step + 1

            scheduler.step()  # Update per epoch

        nn_model.eval()
        with torch.no_grad():
            if batch_partition_size is not None:
                sidx_list = [list(range(s, s + batch_partition_size["val"])) for s in
                             range(0, num_val_samples, batch_partition_size["val"])]

                test_loss_temp = []
                test_anomaly_l2_temp = []
                test_auc_temp = []
                for sidx in sidx_list:
                    val_data_partition = datagen.nw_scenario_subset(val_data, sidx)
                    model_out_partition = nn_model(val_data_partition)
                    model_out_A_partition = model_out_partition[-1]
                    test_loss_temp.append(lossfun.lossfun_simple(val_data_partition, model_out_A_partition, epoch,
                                                  loss_type=loss_type_current, loss_options=loss_options).cpu())
                    test_anomaly_l2_temp.append(evalfun.anomaly_l2(val_data_partition, model_out_A_partition).cpu())
                    test_auc_temp.append(utils.exact_auc(model_out_A_partition[-1], val_data_partition["A"]).cpu())
                test_loss_temp = torch.stack(test_loss_temp).mean()
                test_anomaly_l2_temp = torch.stack(test_anomaly_l2_temp).mean(dim=0)
                test_auc_temp = torch.stack(test_auc_temp).mean()

            else:
                model_out = nn_model(val_data)
                model_out_A = model_out[-1]

                test_loss_temp = lossfun.lossfun_simple(val_data, model_out_A, epoch,
                                              loss_type=loss_type_current, loss_options=loss_options).cpu()
                test_anomaly_l2_temp = evalfun.anomaly_l2(val_data, model_out_A).cpu()
                test_auc_temp = utils.exact_auc(model_out_A[-1], val_data["A"]).cpu()

            test_loss.append(test_loss_temp)
            test_anomaly_l2.append(test_anomaly_l2_temp)
            detector_eval.append({"auc": test_auc_temp})

            if hasattr(nn_model, "lam_val"):
                reg_param["lam"].append(nn_model.lam_val.cpu())
            reg_param["mu"].append(nn_model.mu_val.cpu())
            if hasattr(nn_model, "nu_val"):
                reg_param["nu"].append(nn_model.nu_val.cpu())
            # if hasattr(nn_model, "regf_val"):
            #     reg_param["regf"].append(nn_model.regf_val)
            # if hasattr(nn_model, "regg_val"):
            #     reg_param["regg"].append(nn_model.regf_val)

        print("Test Loss={}".format(test_loss_temp))

        # create checkpoint
        epoch = epoch + 1
        report["running"] = {"epoch": epoch, "step": step,
                             "model_sd": nn_model.state_dict(),
                             "opt_sd": optimizer.state_dict(),
                             "sched_sd": scheduler.state_dict()}

    # reg_param_ll.append(nn_model.layers[-1].get_regularization_parameters(clone=True))

    training_loss_report = torch.stack(training_loss)
    test_loss_report = torch.stack(test_loss)
    test_anomaly_l2_report = torch.stack(test_anomaly_l2)

    report["training_loss"] = training_loss_report
    report["test_loss"] = test_loss_report
    report["test_anomaly_l2"] = test_anomaly_l2_report
    report["grad"] = {"mean": torch.stack(grad_tracking["mean"]),
                      "max": torch.stack(grad_tracking["max"])}
    report["grad_ly"] = {"mean": torch.stack(grad_tracking_ly["mean"], dim=-1),  # stacking steps in -1
                         "max": torch.stack(grad_tracking_ly["max"], dim=-1)}
    if hasattr(nn_model.layers[0], "df_dict"):
        report["df_dict"] = {}
        for k in ["max", "min", "mean"]:
            report["df_dict"][k] = torch.stack([nn_model.layers[l].df_dict[k] for l in range(nn_model.num_layers)])
    if hasattr(nn_model.layers[0], "mu_dict"):
        report["mu_dict"] = {}
        for k in ["max", "min", "mean"]:
            report["mu_dict"][k] = torch.stack([nn_model.layers[l].mu_dict[k] for l in range(nn_model.num_layers)])

    report["model_dict"] = nn_model.state_dict()
    report["opt_dict"] = optimizer.state_dict()
    report["sched_dict"] = optimizer.state_dict()

    report["test_det_eval"] = {}
    for k in detector_eval[0].keys():
        report["test_det_eval"][k] = torch.stack([detector_eval[e][k] for e in range(len(detector_eval))])

    # report["reg_param_ll"] = {}
    # for k in reg_param_ll[0].keys():
    #     report["reg_param_ll"][k] = torch.stack([reg_param_ll[e][k] for e in range(len(reg_param_ll))])

    reg_param_report = {}
    if len(reg_param["lam"]) > 0:
        reg_param_report["lam"] = torch.stack(reg_param["lam"])
    # if len(reg_param["regf"]) > 0:
    #     reg_param_report["regf"] = torch.stack(reg_param["regf"])
    # if len(reg_param["regg"]) > 0:
    #     reg_param_report["regg"] = torch.stack(reg_param["regg"])
    reg_param_report["mu"] = torch.stack(reg_param["mu"])
    if len(reg_param["nu"]) > 0:
        reg_param_report["nu"] = torch.stack(reg_param["nu"])

    report["reg_param"] = reg_param_report

    return report


def crange(start, end, modulo):
    for i in range(start, end):
        yield i % modulo


class MultiStepLRwWarmup(torch.optim.lr_scheduler.MultiStepLR):
    """Implements a MultiStepLR but with a warmup period"""

    warmup_num_steps = None

    def __init__(self, *args, warmup_num_steps=None, **kwargs):
        self.warmup_num_steps = warmup_num_steps
        super().__init__(*args, **kwargs)

    def get_lr(self):
        lrs = super().get_lr()
        if self.warmup_num_steps:
            if self.last_epoch == 0:
                warmup = min(1.0, 1 / self.warmup_num_steps)
            else:
                warmup = min(1.0, (self.last_epoch + 1) / self.warmup_num_steps) / \
                         min(1.0, self.last_epoch / self.warmup_num_steps)
            return [lr * warmup for lr in lrs]
        else:
            return lrs