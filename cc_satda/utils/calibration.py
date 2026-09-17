from typing import List, Tuple, Optional
import torch


def compute_class_priors(dataset_items: List[Tuple[str, int]], num_classes: int = 5, smoothing: float = 1e-4) -> torch.Tensor:
    """
    Computes empirical class frequencies directly from dataset items at runtime:
        pi_c = (count_c + smoothing) / (total + C * smoothing)
    Guarantees mathematically valid priors even in extreme few-shot regimes.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset_items:
        counts[label] += 1.0
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    priors = (counts + smoothing) / (total + num_classes * smoothing)
    return priors


def calibrate_logits(
    logits: torch.Tensor,
    pi_src: Optional[torch.Tensor],
    pi_tgt: Optional[torch.Tensor],
    tau: float = 0.35
) -> torch.Tensor:
    """
    Applies temperature-scaled Bayesian log-prior shift to classification logits:
    
        logits_calibrated = logits + tau * [ log(pi_target) - log(pi_source) ]
        
    This compensates for severe prevalence shifts between academic dermoscopy archives
    (dominant benign moles) and clinic presentation (dominant keratinocyte carcinomas),
    preventing catastrophic collapse on minority malignant classes like Melanoma.
    """
    if tau > 0.0 and pi_src is not None and pi_tgt is not None:
        device = logits.device
        pi_s = pi_src.to(device).clamp(min=1e-6)
        pi_t = pi_tgt.to(device).clamp(min=1e-6)
        shift = torch.log(pi_t) - torch.log(pi_s)
        logits = logits + tau * shift.unsqueeze(0)
    return logits
