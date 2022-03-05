import torch
import torch.nn.functional as F
from torch import nn

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target):

    class_logit = class_logit.reshape(len(label), int(class_logit.shape[0]/len(label)) , -1)
    box_regression = box_regression.reshape(len(label), int(box_regression.shape[0]/len(label)) , -1)

    classifier_loss = []
    box_reg_loss = []
    for i in range(len(label)):
        size = min(class_logit[i].shape[0], label[i].shape[0])
        classifier_loss.append(F.cross_entropy(class_logit[i][:size], label[i][:size]))

        N, num_pos = class_logit[i].shape[0], regression_target[i].shape[0]
        box_regression_i = box_regression[i].reshape(N, -1, 4)
        box_regression_i, label_i = box_regression_i[:num_pos], label[i][:num_pos]
        box_idx = torch.arange(num_pos, device=label_i.device)

        box_reg_loss.append(F.smooth_l1_loss(box_regression_i[box_idx, label_i], regression_target[i], reduction='sum') / N)
    
    classifier_loss = torch.stack(classifier_loss, dim=0).sum(dim=0) / len(classifier_loss)
    box_reg_loss = torch.stack(box_reg_loss, dim=0).sum(dim=0) / len(box_reg_loss)

    return classifier_loss, box_reg_loss


def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    mask_loss = []
    for i in range(len(mask_logit)):
        matched_idx_i = matched_idx[i][:, None].to(proposal[i])
        roi = torch.cat((matched_idx_i, proposal[i]), dim=1)
                
        M = mask_logit[i].shape[-1]
        gt_mask_i = gt_mask[i][:, None].to(roi)
        mask_target = roi_align(gt_mask_i, roi, 1., M, M, -1)[:, 0]

        idx = torch.arange(label[i].shape[0], device=label[i].device)
        mask_loss.append(F.binary_cross_entropy_with_logits(mask_logit[i][idx, label[i]], mask_target))
    mask_loss = torch.stack(mask_loss, dim=0).sum(dim=0) / len(mask_loss)
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections, predict):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        self.predict = predict
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target, index):
        gt_box = target['boxes'][index]
        gt_label = target['labels'][index]
        proposal = torch.cat((proposal, gt_box))
        
        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))
        
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0
        
        return proposal, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        
        boxes = []
        labels = []
        scores = []
        proposal = proposal[0]
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            # if len(torch.unique(keep)) > 1:
            #     print("no keep box")

            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)
            
            box, score = process_box(box, score, image_shape, self.min_size)
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results
    
    def forward(self, feature, proposal, image_shape, target):
        if not self.predict:
            matched_idx, label, regression_target = [], [], []
            for i in range(len(proposal)):
                sing_proposal, sing_matched_idx, sing_label, sing_regression_target = self.select_training_samples(proposal[i], target, i)
                proposal[i] = sing_proposal
                matched_idx.append(sing_matched_idx)
                label.append(sing_label)
                regression_target.append(sing_regression_target)
        
        # box_feature = torch.cat([self.box_roi_pool(feature, proposal[i], image_shape).unsqueeze(0) for i in range(len(proposal))])
        box_feature = torch.cat([self.box_roi_pool(feature, proposal[i], image_shape) for i in range(len(proposal))])
        # box_feature = [self.box_roi_pool(feature, proposal[i], image_shape) for i in range(len(proposal))]
        class_logit, box_regression = self.box_predictor(box_feature)
        
        result, losses = {}, {}
        if not self.predict:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        # classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
        # losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        # # result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape) # if inference
            
        if self.has_mask():
            if not self.predict:
                # num_pos = regression_target.shape[0]
                num_pos = [t.shape[0] for t in regression_target]
                
                # mask_proposal = proposal[:num_pos]
                mask_proposal = [t[:num_pos[idx]] for idx, t in enumerate(proposal)]
                # pos_matched_idx = matched_idx[:num_pos]
                pos_matched_idx = [t[:num_pos[idx]] for idx,t in enumerate(matched_idx)]
                # mask_label = label[:num_pos]
                mask_label = [t[:num_pos[idx]] for idx,t in enumerate(label)]
                
                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''
                mask_proposal_shape = sum([t.shape[0] for t in mask_proposal])
                if mask_proposal_shape == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']               
                
                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                else:
                    mask_proposal = [mask_proposal]

            mask_logit = []
            for i in range(len(mask_proposal)):
                mask_feature = self.mask_roi_pool(feature, mask_proposal[i], image_shape)
                mask_logit.append(self.mask_predictor(mask_feature))
            
            if not self.predict:
                gt_mask = target['masks']
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[0][idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(masks=mask_prob))
                
        return result, losses