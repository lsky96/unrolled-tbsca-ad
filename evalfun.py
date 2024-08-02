import torch
import utils
from reference_algo import network_anomalography_obj
from utils import exact_auc


def unsuper_obj_loss(scenario, P, Q, A, lam, mu):
    # P, Q, A are expected to be (iteration, batch_size, *, *)
    # batch_size handled in lower function
    return network_anomalography_obj(scenario, P, Q, A, lam, mu)


def anomaly_l1(scenario_dict, A_est):
    assert(scenario_dict["batch_size"] == A_est.shape[-3])
    A_true = scenario_dict["A"]
    loss = utils.l1_norm(A_true - A_est, dim=(-2, -1))
    loss = loss.mean(dim=-1)
    return loss


def anomaly_l2(scenario_dict, A_est):
    assert(scenario_dict["batch_size"] == A_est.shape[-3])
    A_true = scenario_dict["A"]
    loss = utils.frob_norm_sq(A_true - A_est, dim=(-2, -1))
    loss = loss.mean(dim=-1)  # batch mean
    return loss


def regularizer(P_est, Q_est, A_est, lam, mu):
    assert(len(A_est.shape) == 4)
    loss = lam * (utils.frob_norm_sq(P_est, dim=(-2, -1)) + utils.frob_norm_sq(Q_est, dim=(-2, -1))) / 2
    loss += mu * utils.l1_norm(A_est, dim=(-2, -1))
    loss = loss.mean(dim=-1)
    return loss


def detector_single_class(scenario_dict, A_est, auc=False):
    """
    :param scenario_dict:
    :param A_est:
    :param auc: Toggle ROC AUC computation. Default False due to computational cost.
    :return:
    """
    """Don't use. Metrics are useless for this threshold."""

    assert (scenario_dict["batch_size"] == A_est.shape[-3])
    A_true = scenario_dict["A"]
    is_anomaly_true = torch.abs(A_true) > 0
    is_anomaly_est = torch.abs(A_est) > 0

    tp = (is_anomaly_true * is_anomaly_est).sum(dim=(-2, -1))
    tn = ((~is_anomaly_true) * (~is_anomaly_est)).sum(dim=(-2, -1))
    fp = ((~is_anomaly_true) * is_anomaly_est).sum(dim=(-2, -1))
    fn = (is_anomaly_true * (~is_anomaly_est)).sum(dim=(-2, -1))
    pd = tp / (tp + fn)  # probability of detection, true positive rate
    pfa = fp / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    f1 = 2 * tp / (2*tp + fp + fn)
    pd = pd.mean(dim=-1)
    pfa = pfa.mean(dim=-1)
    prec = prec.mean(dim=-1)
    acc = acc.mean(dim=-1)
    f1 = f1.mean(dim=-1)

    eval_dict = {"accuracy": acc, "precision": prec, "prob_detection": pd, "prob_false_alarm": pfa, "f1": f1}

    if auc:
        ## ROC AUC Estimate
        A_abs = torch.flatten(A_est.abs(), start_dim=-2)
        labels = torch.flatten(is_anomaly_true, start_dim=-2)
        pd_roc, pfa_roc = roc_sampler(A_abs, labels, num_samples=25)
        auc = roc_auc(pd_roc, pfa_roc).mean(dim=-1)  # mean over batch

        eval_dict["auc"] = auc
    return eval_dict


def roc_sampler(score, label, num_samples=25):
    """

    :param score: (*, N) nonnegative
    :param label: (*, N)
    :param num_samples:
    :return: pd (num_samples, *), pfa (num_samples, *)
    """
    """Score must be positive"""
    score_max = score.max(dim=-1)[0]
    score = score
    label = label

    t_samples = torch.logspace(-4, 0, num_samples-1)[:-1] * score_max.unsqueeze(-1)   # t=amax is just pd=0, pfa=0
    # t_samples = torch.linspace(0.001, 1, num_samples - 1)[:-1] * score_max.unsqueeze(-1)  # t=amax is just pd=0, pfa=0

    p = label.sum(dim=-1)
    n = (~label).sum(dim=-1)

    pd_roc = [torch.ones_like(score_max)]
    pfa_roc = [torch.ones_like(score_max)]
    for idx in range(num_samples-2):
        t = t_samples[..., idx].unsqueeze(-1)
        is_high = score > t

        tp = (label * is_high).sum(dim=-1)
        fp = ((~label) * is_high).sum(dim=-1)

        pd = tp / p  # probability of detection, true positive rate
        pfa = fp / n

        pd_roc.append(pd)
        pfa_roc.append(pfa)

    pd_roc.append(torch.zeros_like(score_max))
    pfa_roc.append(torch.zeros_like(score_max))
    pd_roc = torch.stack(pd_roc)
    pfa_roc = torch.stack(pfa_roc)

    return pd_roc, pfa_roc


def roc_auc(pd_roc, pfa_roc):
    """
    Evaluates the AUC based on the sampled ROC.
    :param pd_roc: (num_samples, *) descending order in dim 0
    :param pfa_roc: pfa (num_samples, *), descending order in dim 0
    :return:
    """

    """NOT USED, NOT PRECISE ENOUGH"""

    assert(pd_roc.shape == pfa_roc.shape)
    num_samples = pd_roc.shape[0]

    areas = (pd_roc[0:-1] + pd_roc[1:]) * (pfa_roc[0:-1] - pfa_roc[1:]) / 2
    auc = areas.sum(dim=0)

    return auc


def roc_curve(A, Agt, num_samples=50, batch_mean=True):
    ## ROC AUC Estimate
    A_abs = torch.flatten(A.abs(), start_dim=-2)
    labels = torch.flatten(Agt != 0, start_dim=-2)
    pd_roc, pfa_roc = roc_sampler(A_abs, labels, num_samples=num_samples)
    if batch_mean:
        pd_roc = pd_roc.mean(dim=-1)
        pfa_roc = pfa_roc.mean(dim=-1)

    return {"pd_roc": pd_roc, "pfa_roc": pfa_roc}