"""Tversky Loss"""
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


def tversky_loss(pred, target, alpha=0.5, beta=0.5):
    num_classes = pred.size(1)
    target_oh = torch.eye(num_classes)[target.squeeze(1)]
    target_oh = target_oh.permute(0, 3, 1, 2).float()
    probs = nn.functional.softmax(pred, dim=1)
    target_oh = target_oh.type(pred.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    inter = torch.sum(probs * target_oh, dims)
    fps = torch.sum(probs * (1 - target_oh), dims)
    fns = torch.sum((1 - probs) * target_oh, dims)
    t = (inter / (inter + (alpha * fps) + (beta * fns))).mean()
    return 1 - t


def tversky_loss_v2(pred, target, alpha=0.5, beta=0.5):
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


def cross_entropy_dice_weighted_loss(
    pred, target, cross_entropy_weight=0.5, tversky_weight=0.5
):
    if cross_entropy_weight + tversky_weight != 1:
        raise ValueError("Cross Entropy weight and Tversky weight should " "sum to 1")
    ce = nn.functional.cross_entropy(pred, target, weight=class_dice(pred, target))
    tv = tversky_loss_v2(pred, target)
    loss = (cross_entropy_weight * ce) + (tversky_weight * tv)
    return loss


def gdl_loss(inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
    num_classes = inputs.shape[-3]
    print(f"{num_classes=}")
    # make target a tensor with num_class dimension as for channels
    target_oh = torch.eye(num_classes)[targets.squeeze()]
    target_oh = target_oh.permute(0, 1, 4, 2, 3).float()
    print(f"{target_oh.shape}")
    sums = 0
    for inp, target in zip(inputs, target_oh):
        sums += torch.abs_(
            torch.abs_(target[:, :, 1:, 1:] - target[:, :, :-1, :-1])
            - torch.abs_(inp[:, :, 1:, 1:] - inp[:, :, :-1, :-1])
        ).sum()

    if reduction == "sum":
        return sums
    elif reduction == "mean":
        return sums / inputs.shape[0]  # average between frames


def l1_gdl(inputs: torch.Tensor, targets: torch.Tensor):
    """L1+Gradient difference loss according to 'Predicting Deeper into the Future of Semantic Segmentation' paper"""
    l1_loss = nn.L1Loss()
    num_classes = inputs.shape[-3]
    # bring Time dim to the front
    # targets = targets.transpose(0, 1)
    # make target a tensor with num_class dimension as for channels
    target_oh = torch.eye(num_classes)[targets.squeeze()]
    # (B, T, H, W, C) -> (B, T, C, H, W)
    target_oh = target_oh.permute(0, 1, 4, 2, 3).float()
    gdl_sum = 0
    l1_sum = 0

    gdl_sum += (
        torch.abs_(
            torch.abs_(target_oh[..., 1:, :] - target_oh[..., :-1, :])
            - torch.abs_(inputs[..., 1:, :] - inputs[..., :-1, :])
        )
        .mean(dim=-3)
        .sum()
        + torch.abs_(
            torch.abs_(target_oh[..., :, 1:] - target_oh[..., :, :-1])
            - torch.abs_(inputs[..., :, 1:] - inputs[..., :, :-1])
        )
        .mean(dim=-3)
        .sum()
    )
    gdl_sum /= inputs.shape[0]  # divide by batch_size
    l1_sum += l1_loss(inputs, target_oh)

    return (gdl_sum + l1_sum) / inputs.shape[1]  # average between frames


def moe_loss(action_dists, speed_pred, actions_gt, speed_gt, loss_coefs):
    """Use NLL for imitation loss and MSE for speed prediction"""
    loglike = action_dists.log_prob(actions_gt)
    nll = -torch.mean(loglike, dim=0)
    mse_fn = nn.MSELoss()
    if len(speed_pred.shape) > 2:
        # speed_loss = 0
        # for i in range(speed_pred.shape[1]):
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
    x = torch.rand(2, 4, 23, 144, 256, requires_grad=True)
    y = torch.rand(2, 4, 144, 256).long()
    y[0, 0, 2, 2] = 21
    l1_gdl(x, y).backward()
    print(x.grad)
    print(l1_gdl(x, y))
    # x = torch.rand(2, 23, 5, 7, requires_grad=True)
    # y = torch.rand(2, 5, 7).long()
    # y[0, 2, 2] = 4
    # print(tversky_loss(x, y) == tversky_loss_v2(x, y))
