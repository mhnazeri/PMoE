"""Collection of loss functions"""
import torch
import torch.nn as nn


def class_dice(pred, target, epsilon=1e-6):
    num_classes = pred.size(1)
    pred_class = torch.argmax(pred, dim=1)
    dice = torch.ones(num_classes, dtype=torch.float, device=pred.device)
    for c in range(num_classes):
        p = pred_class == c
        t = target == c
        inter = (p * t).sum().float() + epsilon
        union = p.sum() + t.sum() + epsilon
        d = 2 * inter / union
        dice[c] = 1 - d
    return dice


def dice_score(pred, target, epsilon=1e-6):
    num_classes = pred.size(1)
    pred_class = torch.argmax(pred, dim=1)
    dice = torch.ones(num_classes, dtype=torch.float, device=pred.device)
    for c in range(num_classes):
        p = pred_class == c
        t = target == c
        inter = (p * t).sum().float() + epsilon
        union = p.sum() + t.sum() + epsilon
        d = 2 * inter / union
        dice[c] = d
    return dice


def tversky_loss(pred, target, alpha=0.5, beta=0.5):
    # num_classes = pred.size(1)
    target_oh = torch.zeros_like(pred, dtype=torch.float, device=pred.device)
    target_oh.scatter_(dim=-3, index=target.unsqueeze(-3), value=1.0)
    probs = nn.functional.softmax(pred, dim=1)
    dims = (0,) + tuple(range(2, target.ndimension()))
    inter = torch.sum(probs * target_oh, dims)
    fps = torch.sum(probs * (1 - target_oh), dims)
    fns = torch.sum((1 - probs) * target_oh, dims)
    t = (inter / (inter + (alpha * fps) + (beta * fns))).mean()
    return 1 - t


def cross_entropy_tversky_weighted_loss(
    pred, target, cross_entropy_weight=0.5, tversky_weight=0.5
):
    if cross_entropy_weight + tversky_weight != 1:
        raise ValueError("Cross Entropy weight and Tversky weight should " "sum to 1")
    ce = nn.functional.cross_entropy(pred, target, weight=class_dice(pred, target))
    tv = tversky_loss(pred, target)
    loss = (cross_entropy_weight * ce) + (tversky_weight * tv)
    return loss


def l1_gdl(inputs: torch.Tensor, targets: torch.Tensor):
    """L1+Gradient difference loss according to 'Predicting Deeper into the Future of Semantic Segmentation' paper

    inputs and targets have shape (B, T, C, H, W)
    """
    l1_loss = nn.L1Loss()
    target_oh = torch.zeros_like(inputs, dtype=torch.float, device=inputs.device)
    target_oh.scatter_(dim=-3, index=targets.unsqueeze(-3), value=1.0)
    inputs_soft = torch.nn.functional.softmax(inputs, dim=-3)
    pad_right = nn.ZeroPad2d((0, 1, 0, 0))
    pad_bottom = nn.ZeroPad2d((0, 0, 0, 1))

    gdl_sum = (
        torch.abs_(
            torch.abs_(pad_bottom(target_oh[:, -1, ...])[..., :, 1:, :] - pad_bottom(target_oh[:, -1, ...])[..., :, :-1, :])
            - torch.abs_(pad_bottom(inputs[:, -1, ...])[..., :, 1:, :] - pad_bottom(inputs[:, -1, ...])[..., :, :-1, :])
        )
        + torch.abs_(
            torch.abs_(pad_right(target_oh[:, -1, ...])[..., :, :, :-1] - pad_right(target_oh[:, -1, ...])[..., :, :, 1:])
            - torch.abs_(pad_right(inputs[:, -1, ...])[..., :, :, :-1] - pad_right(inputs[:, -1, ...])[..., :, :, 1:])
        )
    ).sum(dim=(-2, -1)).mean()

    l1_sum = l1_loss(inputs[:, -1, ...], target_oh[:, -1, ...])

    return l1_sum + gdl_sum


class AutoregressiveCriterion(nn.Module):
    def __init__(self, n_target_frames: int = 1, loss_type: str = 'tversky'):
        """Multi frames loss which backpropagate loss error through time"""
        super().__init__()
        self.n_target_frames = n_target_frames
        self.loss_type = loss_type
        self.loss = None
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        elif loss_type == 'tversky':
            self.loss = cross_entropy_tversky_weighted_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}, supported ones are L1, L2, and tversky")

    def forward(self, inputs, targets):
        """inputs shape is (B, T, C, H, W) where C is 23
          targets shape is (B, T, C, H, W) where C is 1
        """
        assert inputs.size(1) == self.n_target_frames
        assert targets.size(1) == self.n_target_frames

        if self.loss_type != 'tversky':
            target_oh = torch.zeros_like(inputs, dtype=torch.float, device=inputs.device)
            target_oh.scatter_(dim=-3, index=targets.unsqueeze(-3), value=1.0)
            targets = target_oh

        final_loss = 0
        for t in range(self.n_target_frames):
            final_loss += self.loss(inputs[:, t, ...], targets[:, t, ...])

        return final_loss


def moe_loss(action_dists, speed_pred, actions_gt, speed_gt, loss_coefs):
    """Use NLL for imitation loss and MSE for speed prediction"""
    loglike = action_dists.log_prob(actions_gt)
    nll = -torch.mean(loglike, dim=0)
    mse_fn = nn.MSELoss()
    if len(speed_pred.shape) > 2:
        speed_loss = mse_fn(speed_pred, speed_gt.unsqueeze_(1).expand_as(speed_pred))
        speed_loss /= speed_pred.shape[1]
    else:
        speed_loss = mse_fn(speed_pred, speed_gt)

    return loss_coefs[0] * nll + loss_coefs[1] * speed_loss


def punet_loss(actions, speed_pred, actions_gt, speed_gt, loss_coefs):
    """Use L1 loss for imitation loss and L2 loss for speed prediction"""
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    imitation_loss = l1_loss(actions, actions_gt)
    speed_loss = mse_loss(speed_pred, speed_gt)

    return loss_coefs[0] * imitation_loss + loss_coefs[1] * speed_loss


def pmoe_loss(actions, speed_pred, actions_gt, speed_gt, loss_coefs):
    """Use L1 loss for imitation loss. Other variables are dummy for the sake
    of interface consistency
    """
    imitation_loss = nn.functional.l1_loss(actions, actions_gt)

    return imitation_loss


if __name__ == "__main__":
    pass
