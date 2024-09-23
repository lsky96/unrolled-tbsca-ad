import numpy as np
import torch
import pywt

import datagen


def _abilene_flows_considered():
    self_flows = np.arange(12) * 13
    # since ATLAng and ATLA-M5 are at the same location, we remove flows to ATLAng to keep the links 0(1) and 1(2)
    atlang_flows = np.concatenate((np.arange(12, 24), np.arange(12)*12 + 1))

    flows_to_remove = np.concatenate((self_flows, atlang_flows))

    flows_to_use = [f for f in range(144) if f not in flows_to_remove]

    return np.array(flows_to_use)


def _abilene_get_routing_matrix(path):
    values_raw = np.loadtxt(path, skiprows=1, delimiter=" ", usecols=(2,3,4))
    values_raw = values_raw.astype(int)

    temp = values_raw[values_raw[:, 0] <= 30]  # only keep internal links

    R = np.zeros((30, 144))
    for i in range(temp.shape[0]):
        e = temp[i, 0] - 1
        f = temp[i, 1] - 1
        R[e, f] = 1

    flow_idx_considered = _abilene_flows_considered()
    R = R[:, flow_idx_considered]

    return R


def _abilene_read_flow(path):
    values_raw = np.loadtxt(path, delimiter=" ", usecols=list(range(1, 721)))
    real_od_idx = np.arange(144) * 5
    real_od = values_raw[:, real_od_idx]

    flow_idx_considered = _abilene_flows_considered()
    flows = real_od[:, flow_idx_considered]
    Z = flows.T

    return Z


def _smoothed_flow(Z):
    # Z.shape = (F, T)
    # follows Kasai 2016
    # w = pywt.Wavelet("db5")   # Daubechies-5
    num_flows = Z.shape[0]
    num_time = Z.shape[1]

    Z_smoothed = np.zeros_like(Z)
    for f in range(num_flows):
        acoeff = pywt.downcoef("a", Z[f, :], "db5", mode="smooth", level=5)
        Z_smoothed[f, :] = pywt.upcoef("a", acoeff, "db5", level=5, take=num_time)

    return Z_smoothed


def abilene_dataset(rpath, fpaths, sampling_param, preprocessing=False):
    """Preprocessing would be using Wavelets according to Kasai 2016, however, we decided against it."""
    # fpaths - list
    # sampling_param = {anomaly_distr={amplitude, prob, length}}

    ano_amplitude = sampling_param["anomaly_distr"]["amplitude"]
    ano_prob = sampling_param["anomaly_distr"]["prob"]
    ano_len = sampling_param["anomaly_distr"]["len"]
    observation_prob = sampling_param["observation_prob"]

    if "subsampling" in sampling_param:
        subsampling = sampling_param["subsampling"]
    else:
        subsampling = 1
    if "combine_weeks" in sampling_param:
        combine_weeks = sampling_param["combine_weeks"]
    else:
        combine_weeks = 1

    if "anomaly_mixture" in sampling_param:
        ano_mixture = sampling_param["anomaly_mixture"]  # "sum", "mult"
    else:
        ano_mixture = "sum"

    R = _abilene_get_routing_matrix(rpath)
    R = torch.tensor(R.astype(np.float32))
    num_directed_edges = R.shape[0]
    num_timesteps = 2016 // subsampling * combine_weeks
    num_time_seg = 7 * combine_weeks

    batch_size = len(fpaths)
    Z = []
    N = []
    A_ind = []
    for i in range(batch_size):
        Zraw = _abilene_read_flow(fpaths[i])
        Zraw = Zraw / Zraw.mean()  # normalizing flow
        if preprocessing:  # implements preprocessing as in Kasai 2016
            Zsmooth = _smoothed_flow(Zraw)
            Znoise_var = (Zraw - Zsmooth).var(axis=-1)  # see Kasai 2016
            Zflow = Zsmooth
            Znoise = np.random.randn(*Zraw.shape) * np.sqrt(Znoise_var)[:, None]
            # Zfinal = Znoise + Zsmooth
        else:
            Zflow = Zraw

            if subsampling > 1:
                Zflow = Zflow.reshape([*Zflow.shape[:-1], -1, subsampling]).sum(axis=-1)
            Znoise = np.zeros_like(Zflow)

        Zflow = torch.tensor(Zflow.astype(np.float32))
        Znoise = torch.tensor(Znoise.astype(np.float32))
        Z.append(Zflow)
        N.append(R @ Znoise)

        # Anomalies
        ano_indicator = torch.rand_like(Zflow) <= ano_prob
        ano_indicator_idx = torch.nonzero(ano_indicator)
        num_anomalies = ano_indicator_idx.shape[0]
        for ia in range(num_anomalies):
            f, t = ano_indicator_idx[ia]
            ano_indicator[f, t:min([t+ano_len, num_timesteps])] = 1
        # Afinal = ano_indicator * Zfinal * ano_amplitude
        # Afinal = ano_indicator * ano_amplitude
        A_ind.append(ano_indicator)

    """Combines multiple weeks together"""
    if combine_weeks > 1:
        Z = [torch.cat(Z[i:(i+combine_weeks)], dim=-1) for i in range(0, batch_size, combine_weeks)]
        N = [torch.cat(N[i:(i + combine_weeks)], dim=-1) for i in range(0, batch_size, combine_weeks)]
        A_ind = [torch.cat(A_ind[i:(i + combine_weeks)], dim=-1) for i in range(0, batch_size, combine_weeks)]
        batch_size = batch_size // combine_weeks

    Z = torch.stack(Z)
    N = torch.stack(N)
    A_ind = torch.stack(A_ind)

    R = R.expand(batch_size, num_directed_edges, Z.shape[-2])
    # N = torch.zeros((batch_size, num_directed_edges, num_timesteps))

    """Anomaly amp"""
    if isinstance(ano_amplitude, list):

        ano_amplitude_sampled = torch.rand(batch_size, 1, 1) \
                         * (ano_amplitude[1] - ano_amplitude[0]) + ano_amplitude[0]
        # A[anomaly_indicator_pos] = 1
        # A[anomaly_indicator_neg] = -1
        if ano_mixture == "sum":
            A = A_ind * ano_amplitude_sampled
        elif ano_mixture == "mul":
            A = A_ind * ano_amplitude_sampled * Z
        else:
            raise ValueError
    else:
        if ano_mixture == "sum":
            A = A_ind * ano_amplitude
        elif ano_mixture == "mul":
            A = A_ind * ano_amplitude * Z
        # elif ano_mixture == "meanmul":
        #     A = A_ind * ano_amplitude * torch.sqrt(Z * torch.mean(Z, dim=-1, keepdim=True))
        elif ano_mixture == "clipmul":
            # A_ind[Z == 0] = 0
            A = A_ind * ano_amplitude * torch.clip(Z, min=torch.mean(Z, dim=-1, keepdim=True))
        elif ano_mixture == "maxmul":
            Zflowmax = torch.amax(Z, dim=-1, keepdim=True).expand(Z.shape)
            A_ind[Zflowmax == 0] = 0
            A = A_ind * ano_amplitude * Zflowmax
        else:
            raise ValueError

    """Observations"""
    if isinstance(observation_prob, list):
        obs_prob_temp = torch.rand(batch_size, 1, 1) * (observation_prob[1] - observation_prob[0]) + observation_prob[0]
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps) <= obs_prob_temp
    else:
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps) <= observation_prob

    ss_dict = {"batch_size": batch_size, "num_time_seg": num_time_seg, "sampling_param": sampling_param,
               "R": R, "Omega": Omega, "Z": Z, "A": A, "N": N}
    scenario_set = datagen.ScenarioSet(**ss_dict)
    return scenario_set
