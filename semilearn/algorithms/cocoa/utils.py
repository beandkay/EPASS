# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
from semilearn.algorithms.hooks import MaskingHook

class COCOAThresholdingHook(MaskingHook):
    """
    Relative Confidence Thresholding in COCOA
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb_w, logits_x_lb_s, logits_x_ulb_w, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb_w.detach()

        if softmax_x_lb:
            probs_x_lb_w = torch.softmax(logits_x_lb_w.detach(), dim=-1)
            probs_x_lb_s = torch.softmax(logits_x_lb_s.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_lb_w = logits_x_lb_w.detach()
            probs_x_lb_s = logits_x_lb_s.detach()

        max_probs_w, _ = probs_x_lb_w.max(dim=-1)
        p_cutoff_w = max_probs_w.mean() * algorithm.p_cutoff
        max_probs_s, _ = probs_x_lb_s.max(dim=-1)
        p_cutoff_s = max_probs_s.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        mask = torch.logical_and(max_probs.ge(p_cutoff_w), max_probs.ge(p_cutoff_s)).to(max_probs.dtype)
        return mask