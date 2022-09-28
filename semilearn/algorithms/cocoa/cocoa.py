

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import COCOAThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
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

    def forward(self, features, max_probs=None, labels=None, mask=None, reduction="mean", select_matrix=None):
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

class COCOA_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(COCOA_Net, self).__init__()
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
        return {'logits':logits, 'feat':feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

class COCOA(AlgorithmBase):
    """
        COCOA algorithm (https://arxiv.org/abs/2001.07685).

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
        # cocoa specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, dist_align=args.dist_align, ema_p=args.ema_p)
        self.contrastive_criterion = SoftSupConLoss(temperature=self.args.contrastive_T).cuda()
        
    def init(self, p_cutoff, T, hard_label=True, dist_align=True, ema_p=0.999):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.ema_p = ema_p

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(COCOAThresholdingHook(), "MaskingHook")
        super().set_hooks()
        
    def set_model(self): 
        model = super().set_model()
        model = COCOA_Net(model, proj_size=self.args.proj_size)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = COCOA_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model    

    
    def train_step(self, x_lb_w, x_lb_s, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits, feats = outputs['logits'], outputs['feat']
                del inputs, outputs
                logits_x_lb_w, logits_x_lb_s = logits[:num_lb*2].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb*2:].chunk(2)
                features_x_ulb_w, features_x_ulb_s = feats[num_lb*2:].chunk(2)
                del logits, feats
            else:
                outs_x_lb_w = self.model(x_lb_w)
                logits_x_lb_w, _  = outs_x_lb_w['logits'], outs_x_lb_w['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s, _  = outs_x_lb_s['logits'], outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s, features_x_ulb_s = outs_x_ulb_s['logits'], outs_x_ulb_s['feat']                
                # with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w, features_x_ulb_w = outs_x_ulb_w['logits'], outs_x_ulb_w['feat']

            sup_loss = ce_loss(logits_x_lb_w, y_lb, reduction='mean') + ce_loss(logits_x_lb_s, y_lb, reduction='mean')
            probs_x_lb_w = torch.softmax(logits_x_lb_w.detach(), dim=-1)
            probs_x_lb_s = torch.softmax(logits_x_lb_s.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb_w)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb_w=probs_x_lb_w, logits_x_lb_s=probs_x_lb_s, logits_x_ulb_w=probs_x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # calculate loss
            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            with torch.no_grad():
                max_probs = torch.max(probs_x_ulb_w, dim=-1)[0]
                
            features_ulb = torch.cat([features_x_ulb_w.unsqueeze(1), features_x_ulb_s.unsqueeze(1)], dim=1)
            contrastive_loss = self.contrastive_criterion(features_ulb, max_probs, pseudo_label) + \
                self.contrastive_criterion(features_ulb)
            
            total_loss = sup_loss + self.lambda_u * unsup_loss + contrastive_loss
            
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/contrastive_loss'] = contrastive_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]