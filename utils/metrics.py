"""
metrics.py
(1) PEHE: Precision in Estimation of Heterogeneous Effect
(2) ATE: Average Treatment Effect
"""

# Necessary packages
import numpy as np
import torch
from torch import autograd
from torch.autograd import  Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor

def hsic(emb1, emb2, dim, device):
    J = torch.eye(dim).to(device) - (1/dim) * torch.ones(dim, dim).to(device)
    K1 = torch.mm(emb1, emb1.t()).to(device)
    K2 = torch.mm(emb2, emb2.t()).to(device)
    JK1 = torch.mm(J, K1).to(device)
    JK2 = torch.mm(J, K2).to(device)
    HSIC = torch.trace(torch.mm(JK1, JK2)).to(device)
    return HSIC / (dim - 1)**2

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    nums = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:nums]
    fake_samples = fake_samples[:nums]
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def PEHE(y, y_hat):
    """Compute Precision in Estimation of Heterogeneous Effect.

    Args:
      - y: potential outcomes
      - y_hat: estimated potential outcomes

    Returns:
      - PEHE_val: computed PEHE
    """
    PEHE_val = np.sqrt(np.mean(np.abs((y[:, 1] - y[:, 0]) - (y_hat[:, 1] - y_hat[:, 0]))))
    #PEHE_val = np.sqrt(np.mean(np.square((y[:,1] - y[:,0]) - (y_hat[:,1] - y_hat[:,0]) )))
    return PEHE_val


def ATE(y, y_hat):
    """Compute Average Treatment Effect.

    Args:
      - y: potential outcomes
      - y_hat: estimated potential outcomes

    Returns:
      - ATE_val: computed ATE
    """
    ATE_val = np.abs(np.mean(y[:, 1] - y[:, 0]) - np.mean(y_hat[:, 1] - y_hat[:, 0]))
    return ATE_val
def abs_ate(mu1,mu0, ypred1, ypred0):
    return np.abs(np.mean(ypred1 - ypred0) - np.mean(mu1-mu0))
def pehe_val(mu1,mu0,ypred1, ypred0):
    return np.sqrt(np.mean(np.square((mu1 - mu0) - (ypred1 - ypred0))))
def PEHE_IHDP(y0,y1,y_hat_0,y_hat_1):
    PEHE=np.sqrt(np.mean(np.abs((y1 - y0) -(y_hat_1-y_hat_0))))
    #PEHE = np.sqrt(np.mean(np.square((y0 - y1) - (y_hat_0 - y_hat_1))))
    return PEHE

def ATE_IHDP(y0,y1,y_hat_0,y_hat_1):
    ATE=np.abs(np.mean(y1 - y0) - np.mean(y_hat_1-y_hat_0))
    return ATE
# third party

def sqrt_PEHE(po: torch.Tensor, hat_te: torch.Tensor) -> torch.Tensor:
    """
    Precision in Estimation of Heterogeneous Effect(PyTorch version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        po: expected outcome.
        hat_te: estimated outcome.
    """
    po = torch.Tensor(po)
    hat_te = torch.Tensor(hat_te)
    return torch.sqrt(torch.mean(((po[:, 1] - po[:, 0]) - hat_te) ** 2))


def abs_error_ATE(po: torch.Tensor, hat_te: torch.Tensor) -> torch.Tensor:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        po: expected outcome.
        hat_te: estimated outcome.
    """
    po = torch.Tensor(po)
    hat_te = torch.Tensor(hat_te)
    return torch.abs(torch.mean(po[:, 1] - po[:, 0]) - torch.mean(hat_te))
def policy_val(t, yf, eff_pred):
    """ Computes the value of the policy defined by predicted effect """

    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred>0
    policy=policy.squeeze()
    t=t.squeeze()
    treat_overlap = (policy==t)*(t>0)
    control_overlap = (policy==t)*(t<1)
    if np.sum(treat_overlap)==0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap)==0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit*treat_value + (1-pit)*control_value

    policy_curve = []

    return policy_value, policy_curve
def Jobs_metric(yf,yf_p,ycf_p,t,e):
    att = np.mean(yf[t > 0]) - np.mean(yf[(1 - t + e) > 1])
    eff_pred = ycf_p - yf_p;
    eff_pred[t > 0] = -eff_pred[t > 0];

    ate_pred = np.mean(eff_pred[e > 0])
    atc_pred = np.mean(eff_pred[(1 - t + e) > 1])

    att_pred = np.mean(eff_pred[(t + e) > 1])
    bias_att = att_pred - att
    policy_value, policy_curve = \
            policy_val(t[e > 0], yf[e > 0], eff_pred[e > 0])
    if(policy_value==1):
        policy_value=0
    return np.abs(bias_att),1-policy_value
def ATT(yf,yf_p,ycf_p,t,e):
    att = np.mean(yf[t > 0]) - np.mean(yf[(1 - t + e) > 1])
    eff_pred = ycf_p - yf_p
    eff_pred[t > 0] = -eff_pred[t > 0]
    att_pred = np.mean(eff_pred[(t + e) > 1])
    bias_att = att_pred - att
    return np.abs(bias_att)
def ROL(yf,yf_p,ycf_p,t,e):
    att = np.mean(yf[t > 0]) - np.mean(yf[(1 - t + e) > 1])
    eff_pred = ycf_p - yf_p
    eff_pred[t > 0] = -eff_pred[t > 0]
    att_pred = np.mean(eff_pred[(t + e) > 1])
    bias_att = att_pred - att
    return bias_att
if __name__ == '__main__':
    pass