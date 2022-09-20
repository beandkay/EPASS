

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.multiprocessing import reductions
import torch.nn as nn
import torch.nn.functional as F
from semilearn.algorithms.algorithmbase import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, consistency_loss,  SSL_Argument, str2bool

class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, max_probs, labels=None, mask=None, reduction="mean", select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and select_matrix is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            mask = mask.mul(score_mask) * select_matrix

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #max_probs = max_probs.reshape((batch_size,1))
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs,max_probs.T)
            mask = mask.mul(score_mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if reduction == "mean":
            loss = loss.mean()

        return loss

class CAGUL_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(CAGUL_Net, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_planes, proj_size)
        ])
        
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return logits, feat_proj 

class CAGUL(AlgorithmBase):
    """
        CAGUL algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # cagul specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, dist_align=args.dist_align, ema_p=args.ema_p)
        # warp model
        backbone = self.model
        self.model = CAGUL_Net(backbone, proj_size=self.args.proj_size)
        self.ema_model = CAGUL_Net(self.ema_model, proj_size=self.args.proj_size)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.contrastive_criterion = SoftSupConLoss(temperature=self.args.contrastive_T).cuda()
        
    def init(self, T, p_cutoff, hard_label=True, dist_align=True, ema_p=0.999):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.ema_p = ema_p
        self.lb_prob_t = torch.ones((self.args.num_classes)).cuda(self.args.gpu) / self.args.num_classes
        self.ulb_prob_t = torch.ones((self.args.num_classes)).cuda(self.args.gpu) / self.args.num_classes
    
    @torch.no_grad()
    def update_prob_t(self, lb_probs, ulb_probs):
        if self.args.distributed and self.args.world_size > 1:
            lb_probs = self.concat_all_gather(lb_probs)
            ulb_probs = self.concat_all_gather(ulb_probs)
        
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        lb_prob_t = lb_probs.mean(0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

    @torch.no_grad()
    def distribution_alignment(self, probs):
        # da
        probs = probs * (1e-6 + self.lb_prob_t) / (1e-6 + self.ulb_prob_t)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()
    
    def train_step(self, x_lb_w, x_lb_s, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                logits, feats = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s, logits_x_ulb_w, logits_x_ulb_s = logits.chunk(4)
                _, _, features_x_ulb_w, features_x_ulb_s = feats.chunk(4)
            else:
                logits_x_lb_w, _ = self.model(x_lb_w)
                logits_x_lb_s, _ = self.model(x_lb_s)
                logits_x_ulb_s, features_x_ulb_s = self.model(x_ulb_s)
                with torch.no_grad():
                    logits_x_ulb_w, features_x_ulb_w = self.model(x_ulb_w)

            sup_loss = ce_loss(logits_x_lb_w, y_lb, reduction='mean') + ce_loss(logits_x_lb_s, y_lb, reduction='mean')
            probs_x_lb_w = torch.softmax(logits_x_lb_w.detach(), dim=-1)
            probs_x_lb_s = torch.softmax(logits_x_lb_s.detach(), dim=-1)
            max_probs_lb_w, _ = probs_x_lb_w.max(dim=-1)
            max_probs_lb_s, _ = probs_x_lb_s.max(dim=-1)
            max_probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # update 
            self.update_prob_t(max_probs_lb_w, max_probs_x_ulb_w)

            # distribution alignment
            if self.dist_align:
                max_probs_x_ulb_w = self.distribution_alignment(max_probs_x_ulb_w)

            # compute mask
            with torch.no_grad():
                max_probs = torch.max(max_probs_x_ulb_w, dim=-1)[0]
                p_cutoff_w = max_probs_lb_w.mean() * self.p_cutoff
                p_cutoff_s = max_probs_lb_s.mean() * self.p_cutoff
                mask = torch.logical_and(max_probs.ge(p_cutoff_w), max_probs.ge(p_cutoff_s)).to(max_probs.dtype)
            
            unsup_loss, pseudo_label = consistency_loss(logits_x_ulb_s,
                                             max_probs_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
                                             T=self.T,
                                             mask=mask)
            
            features_ulb = torch.cat([features_x_ulb_w.unsqueeze(1), features_x_ulb_s.unsqueeze(1)], dim=1)
            contrastive_loss = self.contrastive_criterion(features_ulb, max_probs, pseudo_label) + \
                self.contrastive_criterion(features_ulb, pseudo_label)
            
            total_loss = sup_loss + self.lambda_u * unsup_loss + contrastive_loss

        self.parameter_update(total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/contrastive_loss'] = contrastive_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['ulb_prob_t'] = self.ulb_prob_t.cpu()
        save_dict['lb_prob_t'] = self.lb_prob_t.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.ulb_prob_t = checkpoint['ulb_prob_t'].cuda(self.args.gpu)
        self.lb_prob_t = checkpoint['lb_prob_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
