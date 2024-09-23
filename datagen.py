from copy import deepcopy
import numpy as np
import torch
import networkx as nx
from itertools import permutations
from config import DEBUG, FP_DTYPE
import matplotlib.pyplot as plt

FP_DTYPE = torch.float32  # no reason for inreased precision here


class ScenarioSet():
    stype = "ROmPQAN"  # realization variables that are stored. All others are computed on the fly. ("ROmPQAN", "ROmZAN")
    device = None
    data = None
    temp_data = None

    def __init__(self, **kwargs):
        # self.data = {"batch_size": kwargs["batch_size"], "graph_param": None, "sampling_param": None,
        #              "num_time_seg": None}
        self.data = {"batch_size": kwargs.pop("batch_size"), "graph_param": None, "sampling_param": None,
                     "num_time_seg": None}
        self.temp_data = {}

        if "graph_param" in kwargs.keys():
            # self.data["graph_param"] = kwargs["graph_param"]
            self.data["graph_param"] = kwargs.pop("graph_param")

        if "sampling_param" in kwargs.keys():
            # self.data["sampling_param"] = kwargs["sampling_param"]
            self.data["sampling_param"] = kwargs.pop("sampling_param")

        if "num_time_seg" in kwargs.keys():
            # self.data["num_time_seg"] = kwargs["num_time_seg"]
            self.data["num_time_seg"] = kwargs.pop("num_time_seg")

        for t in self._valid_stypes():
            if all(elem in kwargs.keys() for elem in self._stype2storedvals(t)):
                self.stype = t
                for k in self._stype2storedvals(t):
                    # self.data[k] = kwargs[k]
                    self.data[k] = kwargs.pop(k)
                break

        if len(kwargs) > 0:
            raise ValueError("Unknown keys {}.".format(list(kwargs.keys())))

        self.device = torch.device("cpu")  # This is hopefully consistent with the data because I do not check.

    def __getitem__(self, item):
        if item in self._stype2storedvals(self.stype) or item in ["batch_size", "graph_param", "sampling_param",
                                                                  "num_time_seg"]:
            return self.data[item]

        d = self.data
        if self.stype == "ROmPQAN":
            # if item == "Z":
            #     return d["P"] @ d["Q"].mT  # P, Q is not in the sense of the model but as factorization of Z
            if item == "X":
                return d["R"] @ self["Z"]
            elif item == "Y":
                Y = d["Omega"] * (d["R"] @ (d["P"] @ d["Q"].mT + d["A"]) + d["N"])
                return Y
            else:
                raise ValueError
        elif self.stype == "ROmZAN":
            if item == "Y":
                Y = d["Omega"] * (d["R"] @ (d["Z"] + d["A"]) + d["N"])
                return Y
            elif item == "X":
                return d["R"] @ d["Z"]
            else:
                raise ValueError

    def __setitem__(self, key, value):
        if key in self._stype2storedvals(self.stype) or key in ["batch_size", "graph_param", "sampling_param",
                                                                "num_time_seg"]:
            self.data[key] = value
        else:
            raise ValueError("Key cannot be set for stype.")

    def return_subset(self, idxs):
        subset_data = {}
        subset_data["batch_size"] = len(idxs)
        if "num_time_seg" in self.data:
            subset_data["num_time_seg"] = deepcopy(self.data["num_time_seg"])
        subset_data["graph_param"] = deepcopy(self.data["graph_param"])
        subset_data["sampling_param"] = deepcopy(self.data["sampling_param"])

        for k in self._stype2storedvals(self.stype):
            if k == "exception":
                pass  # no exception yet
            else:
                subset_data[k] = self.data[k][idxs]

        subset = ScenarioSet(**subset_data)
        subset.device = self.device  # I hope it is consistent!
        return subset

    def _stype2storedvals(self, type):
        if type == "ROmPQAN":
            return ["R", "Omega", "P", "Q", "A", "N"]
        elif type == "ROmZAN":
            return ["R", "Omega", "Z", "A", "N"]
        else:
            raise ValueError

    def _valid_stypes(self):
        return ["ROmZAN", "ROmPQAN"]

    def to(self, device):
        for k in self._stype2storedvals(self.stype):
            self.data[k] = self.data[k].to(device)


def generate_synthetic_nw_scenarios(batch_size=1, num_timesteps=10, num_time_seg=1, graph_param=None,
                                    sampling_param=None, nflow_version="exp+exp"):
    if sampling_param is None:
        sampling_param = {}
    if graph_param is None:
        graph_param = {}

    num_nodes = graph_param["num_nodes"]
    num_edges = graph_param["num_edges"]
    min_distance = graph_param["min_distance"]

    flow_distr = sampling_param["flow_distr"]  # {"rank", "scale"} rank can be number or interval
    flow_structure = flow_distr["structure"] if "structure" in flow_distr else "svd"
    anomaly_distr = {"prob": None, "amplitude": torch.tensor(1.0, dtype=FP_DTYPE)}
    anomaly_distr = anomaly_distr | sampling_param[
        "anomaly_distr"]  # {"prob", "amplitude"}  # the amplitudes should be positive (number or interval)
    noise_distr = sampling_param["noise_distr"]  # {"variance"}
    observation_prob = sampling_param["observation_prob"]  # float or list of two floats

    trials = 0
    i_scenario = 0
    scenario_graphs = []

    while i_scenario < batch_size:

        """Graph Generation"""
        if trials > 3 * batch_size:
            raise RuntimeError("Passed graph parameters yield connected graph with too low probability.")

        pos = {i: torch.rand(2).numpy() for i in range(num_nodes)}
        G = nx.random_geometric_graph(num_nodes, min_distance, pos=pos)

        # ensure that the graph has the desired number of edges

        while G.number_of_edges() > num_edges:
            # remove excess edges randomly
            excess_edges = G.number_of_edges() - num_edges
            edge_indices = np.arange(len(G.edges()))  # create an array of edge indices
            edges_to_remove = np.random.choice(edge_indices, size=excess_edges, replace=False)
            G.remove_edges_from([list(G.edges())[i] for i in edges_to_remove])

        if nx.is_connected(G) and G.number_of_edges() == num_edges:  # also prevent that it has lower number of edges
            # print(G.number_of_edges())
            scenario_graphs.append(G)
            i_scenario += 1

        trials += 1

    """Routing Matrix Generation"""
    num_flows = num_nodes * (num_nodes - 1)
    num_directed_edges = 2 * num_edges
    R = torch.zeros(batch_size, num_directed_edges, num_flows)
    od_pairs = list(permutations(list(range(num_nodes)), 2))
    # For each OD pair, calculate the minimum hop count route and add it to the routing matrix
    for i_scenario in range(batch_size):
        edges = list(scenario_graphs[i_scenario].edges())
        edges_digraph = [*edges, *[e[::-1] for e in edges]]
        for i_flow, od in enumerate(od_pairs):
            route = nx.shortest_path(scenario_graphs[i_scenario], od[0], od[1])
            for i_dir_edge, edge in enumerate(edges_digraph):
                # Check if the edge is present in the path
                # if set(edge).issubset(path):
                if _directed_edge_in_path(route, edge):
                    # If so, set the corresponding entry in the matrix to 1
                    R[i_scenario][i_dir_edge][i_flow] = 1

    """Flow Generation"""

    def generate_nflow_svd(flow_distr, nflow_ver="exp+exp"):
        rank_cfg = flow_distr["rank"]
        scale_cfg = flow_distr["scale"] if "scale" in flow_distr else None
        # scale_het_cfg = flow_distr["scale_het"] if "scale_het" in flow_distr else None  # heterogeneous distr in batches (default), flows etc.
        if isinstance(rank_cfg, list):
            flow_rank_min = rank_cfg[0]
            flow_rank_max = rank_cfg[1]
            flow_ranks = torch.randint(low=flow_rank_min, high=flow_rank_max + 1, size=(batch_size,))
        else:
            flow_rank_max = flow_distr["rank"]
            flow_ranks = torch.tensor(flow_distr["rank"])

        if nflow_ver == "mardani":
            gauss_distr_var = 1 / num_flows
            U = torch.randn(batch_size, num_flows, flow_rank_max, dtype=FP_DTYPE) * torch.sqrt(
                torch.tensor(gauss_distr_var))
            W = torch.randn(batch_size, num_timesteps, flow_rank_max, dtype=FP_DTYPE)
        elif nflow_ver == "exp+uni":
            uniform_distr_var = 1 / num_flows
            U = torch.rand(batch_size, num_flows, flow_rank_max) * torch.sqrt(torch.tensor(12 * uniform_distr_var))

            qsampler = torch.distributions.exponential.Exponential(torch.tensor(1.0))  # exponential with variance 1
            W = qsampler.sample(sample_shape=(batch_size, num_timesteps, flow_rank_max))

        elif nflow_ver == "abs_gaussian":
            gauss_distr_var = 1 / flow_ranks.unsqueeze(-1).unsqueeze(-1)
            U = torch.randn(batch_size, num_flows, flow_rank_max, dtype=FP_DTYPE).abs() * torch.sqrt(
                torch.tensor(gauss_distr_var))
            W = torch.randn(batch_size, num_timesteps, flow_rank_max, dtype=FP_DTYPE).abs()

        elif nflow_ver == "exp+exp":
            distr_scale = 1 / flow_ranks.unsqueeze(-1).unsqueeze(-1)
            qsampler = torch.distributions.exponential.Exponential(torch.tensor(1.0))
            U = qsampler.sample(sample_shape=(batch_size, num_flows, flow_rank_max)) * torch.tensor(distr_scale)
            W = qsampler.sample(sample_shape=(batch_size, num_timesteps, flow_rank_max))

        else:
            raise ValueError

        # Sampling for random rank
        if isinstance(rank_cfg, list):
            # flow_ranks = torch.randint(low=flow_rank_min, high=flow_rank_max+1, size=(batch_size,))
            rank_mask = torch.arange(flow_rank_max).expand(batch_size, -1) < flow_ranks.unsqueeze(-1)
            U = U * rank_mask.unsqueeze(-2)
            W = W * rank_mask.unsqueeze(-2)

        fscale_cfg = None
        tscale_cfg = None
        if scale_cfg is not None:
            if isinstance(scale_cfg, list):
                if "het_flows" in scale_cfg:  # must be in [2]
                    fscale_cfg = torch.rand(batch_size, num_flows, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[0]
                else:
                    fscale_cfg = torch.rand(batch_size, 1, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[0]
                U = U * fscale_cfg

                if "het_time" in scale_cfg:
                    tscale_cfg = torch.rand(batch_size, num_timesteps, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[0]
                    W = W * tscale_cfg
            else:
                U = U * scale_cfg

        sc = torch.ones(batch_size, num_flows, num_timesteps)
        if fscale_cfg is not None:
            sc = sc * fscale_cfg
        if tscale_cfg is not None:
            sc = sc * tscale_cfg.movedim(-1, -2)

        return U, W, sc

    def generate_nflow_cpd(flow_distr, num_time_seg):
        rank_cfg = flow_distr["rank"]
        scale_cfg = flow_distr["scale"] if "scale" in flow_distr else None
        assert (num_timesteps % num_time_seg == 0)
        num_timesteps1 = num_timesteps // num_time_seg
        num_timesteps2 = num_time_seg

        if isinstance(rank_cfg, list):
            flow_rank_min = rank_cfg[0]
            flow_rank_max = rank_cfg[1]
            flow_ranks = torch.randint(low=flow_rank_min, high=flow_rank_max + 1, size=(batch_size,))
        else:
            flow_rank_max = flow_distr["rank"]
            flow_ranks = torch.tensor(flow_distr["rank"])

        distr_scale = 1 / flow_ranks.unsqueeze(-1).unsqueeze(-1)
        qsampler = torch.distributions.exponential.Exponential(torch.tensor(1.0))
        U = qsampler.sample(sample_shape=(batch_size, num_flows, flow_rank_max)) * distr_scale
        V = qsampler.sample(sample_shape=(batch_size, num_timesteps1, flow_rank_max))
        W = qsampler.sample(sample_shape=(batch_size, num_timesteps2, flow_rank_max))

        # Sampling for random rank
        if isinstance(rank_cfg, list):
            # flow_ranks = torch.randint(low=flow_rank_min, high=flow_rank_max+1, size=(batch_size,))
            rank_mask = torch.arange(flow_rank_max).expand(batch_size, -1) < flow_ranks.unsqueeze(-1)
            U = U * rank_mask.unsqueeze(-2)
            V = V * rank_mask.unsqueeze(-2)
            W = W * rank_mask.unsqueeze(-2)


        fscale_cfg = None
        tscale_cfg = None
        # tscale1_cfg = None
        # tscale2_cfg = None
        if scale_cfg is not None:
            if isinstance(scale_cfg, list):
                if "het_flows" in scale_cfg:
                    # print("SCALE HETEROGENEOUS OVER FLOWS")
                    fscale_cfg = torch.rand(batch_size, num_flows, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[0]
                else:
                    fscale_cfg = torch.rand(batch_size, 1, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[0]
                U = U * fscale_cfg
                if "het_time" in scale_cfg:
                    tscale1_cfg = torch.rand(batch_size, num_timesteps1, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[
                        0]
                    V = V * tscale1_cfg
                    # tscale1_cfg = tscale1_cfg.tile((num_timesteps2, 1))  # expand to size of unfolded tensor

                    tscale2_cfg = torch.rand(batch_size, num_timesteps2, 1) * (scale_cfg[1] - scale_cfg[0]) + scale_cfg[
                        0]
                    W = W * tscale2_cfg

                    tscale_cfg = tscale1_cfg.tile((num_timesteps2, 1)) * \
                                 torch.repeat_interleave(tscale2_cfg, num_timesteps1, dim=1)
            else:
                U = U * scale_cfg

        sc = torch.ones(batch_size, num_flows, num_timesteps)
        if fscale_cfg is not None:
            sc = sc * fscale_cfg
        if tscale_cfg is not None:
            sc = sc * tscale_cfg.movedim(-1, -2)

        # packing V and W
        VW = (V.unsqueeze(-3) * W.unsqueeze(-2)).reshape(batch_size, num_timesteps, flow_rank_max)

        return U, VW, sc

    if flow_structure == "svd":
        P, Q, nflow_scale = generate_nflow_svd(flow_distr, nflow_ver=nflow_version)
        Z = P @ Q.mT
    elif flow_structure == "cpd":
        P, Q, nflow_scale = generate_nflow_cpd(flow_distr, num_time_seg)
        Z = P @ Q.mT
    else:
        raise ValueError

    """Anomaly Generation"""
    if isinstance(anomaly_distr["prob"], list):
        assert (anomaly_distr["prob"][1] > anomaly_distr["prob"][0])
        A_seed = torch.rand(batch_size, 1, 1, dtype=FP_DTYPE) * (anomaly_distr["prob"][1] - anomaly_distr["prob"][0]) + \
                 anomaly_distr["prob"][0]
    else:
        A_seed = anomaly_distr["prob"]

    A = torch.zeros(batch_size, num_flows, num_timesteps, dtype=FP_DTYPE)
    # {1,-1} anomalies
    temp_rand = torch.rand_like(A)
    anomaly_indicator_pos = temp_rand <= A_seed / 2  # anomalies with value +1
    anomaly_indicator_neg = temp_rand >= (1 - A_seed / 2)  # anomalies with value -1

    if isinstance(anomaly_distr["amplitude"], list):
        if "het_flows" in anomaly_distr["amplitude"]:
            ano_amplitude = torch.rand(batch_size, num_flows, 1, dtype=FP_DTYPE) \
                            * (anomaly_distr["amplitude"][1] - anomaly_distr["amplitude"][0]) + \
                            anomaly_distr["amplitude"][
                                0]
        elif "het_mirror_normal" in anomaly_distr["amplitude"]:
            ano_amplitude = nflow_scale * anomaly_distr["amplitude"][0]
        else:
            ano_amplitude = torch.rand(batch_size, 1, 1, dtype=FP_DTYPE) \
                            * (anomaly_distr["amplitude"][1] - anomaly_distr["amplitude"][0]) + \
                            anomaly_distr["amplitude"][
                                0]
        A[anomaly_indicator_pos] = 1
        A[anomaly_indicator_neg] = -1
        A = A * ano_amplitude
    else:
        A[anomaly_indicator_pos] = anomaly_distr["amplitude"]
        A[anomaly_indicator_neg] = -anomaly_distr["amplitude"]

    """Noise Generation"""
    # noise_het_cfg = noise_distr["het"] if "het" in noise_distr else None
    if isinstance(noise_distr["variance"], list) and "het_mirror_normal" in noise_distr["variance"]:  # noise in Z
        noise_std = nflow_scale * torch.sqrt(torch.tensor(noise_distr["variance"][0]))
        ZN = noise_std * torch.randn_like(Z)
        Z = Z + ZN
        N = torch.zeros(batch_size, num_directed_edges, num_timesteps, dtype=FP_DTYPE)
        # P, Q = None, None  # loses normal flow GT for this case, but is for sanity and storage
    else:  # noise in N
        if isinstance(noise_distr["variance"], list):
            if "het_edges" in noise_distr["variance"]:
                noise_var = torch.rand(batch_size, num_directed_edges, 1, dtype=FP_DTYPE) * (
                            noise_distr["variance"][1] - noise_distr["variance"][0]) + noise_distr["variance"][0]
            else:
                noise_var = torch.rand(batch_size, 1, 1, dtype=FP_DTYPE) * (
                            noise_distr["variance"][1] - noise_distr["variance"][0]) + noise_distr["variance"][0]

        else:
            noise_var = torch.tensor(noise_distr["variance"])
        N = torch.sqrt(noise_var) * torch.randn(batch_size, num_directed_edges, num_timesteps, dtype=FP_DTYPE)

    # """Ensure Z+A >= 0"""
    # undershoot = torch.clip(Z + A, max=0)
    # A = A - undershoot

    """Observations"""
    if isinstance(observation_prob, list):
        obs_prob_temp = torch.rand(batch_size, 1, 1) * (observation_prob[1] - observation_prob[0]) + observation_prob[0]
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps, dtype=FP_DTYPE) <= obs_prob_temp
    else:
        Omega = torch.rand(batch_size, num_directed_edges, num_timesteps, dtype=FP_DTYPE) <= observation_prob

    scenario_set = ScenarioSet(batch_size=batch_size,
                               num_time_seg=num_time_seg,
                               graph_param=graph_param,
                               sampling_param=sampling_param,
                               R=R, Omega=Omega, Z=Z, A=A, N=N)

    return scenario_set


def _directed_edge_in_path(path, edge):
    for i in range(len(path) - 1):
        subpath = path[i:(i + 2)]
        if edge[0] == subpath[0] and edge[1] == subpath[1]:
            return True
    return False


def show_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=1000, with_labels=True, width=0.3)
    plt.show(block=None)
    # print("test")


def nw_scenario_observation(scenario_set):
    Y = scenario_set["Y"]
    R = scenario_set["R"]
    Omega = scenario_set["Omega"]
    return Y, R, Omega


def nw_scenario_subset(scenario_set, indices):
    if isinstance(scenario_set, ScenarioSet):
        return scenario_set.return_subset(indices)
    else:
        new_scenario_set = {}
        new_scenario_set["batch_size"] = len(indices)
        if "num_time_seg" in scenario_set.keys():
            new_scenario_set["num_time_seg"] = scenario_set["num_time_seg"]
        if "graph_param" in scenario_set.keys():
            new_scenario_set["graph_param"] = scenario_set["graph_param"]
        if "sampling_param" in scenario_set.keys():
            new_scenario_set["sampling_param"] = scenario_set["sampling_param"]
        new_scenario_set["Y"] = scenario_set["Y"][indices]
        new_scenario_set["R"] = scenario_set["R"][indices]
        new_scenario_set["Omega"] = scenario_set["Omega"][indices]
        new_scenario_set["Z"] = scenario_set["Z"][indices]
        new_scenario_set["A"] = scenario_set["A"][indices]
        new_scenario_set["N"] = scenario_set["N"][indices]

    return new_scenario_set
