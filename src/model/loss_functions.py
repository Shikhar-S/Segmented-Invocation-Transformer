import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingKLLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100,reduction = 'sum'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingKLLoss, self).__init__()
        self.reduction  =  reduction

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes must be in log space
        target (LongTensor): batch_size
        """
        if output.dim() == 2 and target.dim() == 1:
            model_prob = self.one_hot.repeat(target.size(0), 1).to(output.device)
            model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
            model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        elif output.dim() == 3 and target.dim() == 2:
            model_prob = self.one_hot.unsqueeze(1).repeat(target.size(0),target.size(1),1).to(output.device)
            model_prob.scatter_(2, target.unsqueeze(-1), self.confidence)
            model_prob.masked_fill_((target == self.ignore_index).unsqueeze(-1), 0)
            model_prob = model_prob.transpose(1,2)
        else:
            assert False
        return F.kl_div(output, model_prob, reduction = self.reduction)

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction="mean", weight=None, ignore_index = 0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight    = weight
        self.ignore_index = ignore_index

    def reduce_loss(self, loss,nelements):
        return loss.sum()/nelements if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, log_preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(log_preds.device)

        n = log_preds.size(1)
        # log_preds = F.log_softmax(preds, dim=1)
        loss = -log_preds.sum(dim=1)
        ignore_mask = target == self.ignore_index
        loss.masked_fill_(ignore_mask,0)
        loss = self.reduce_loss(loss,ignore_mask.sum())

        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight, ignore_index= self.ignore_index
        )
        return self.linear_combination(loss / n, nll)