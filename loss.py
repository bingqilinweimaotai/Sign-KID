# coding: utf-8
"""
Module to implement training loss
"""

from torch import nn, Tensor
import torch

def getSkeletalModelStructure():
    return (
        # head
        (0, 1),

        # left shoulder
        (1, 2),

        # left arm
        (2, 3),
        # (3, 4, 3),
        # Changed to avoid wrist, go straight to hands
        (3, 29),

        # right shoulder
        (1, 5),

        # right arm
        (5, 6),
        # (6, 7, 3),
        # Changed to avoid wrist, go straight to hands
        (6, 8),

        # left hand - wrist
        # (7, 8, 4),

        # left hand - palm
        (8, 9),
        (8, 13),
        (8, 17),
        (8, 21),
        (8, 25),

        # left hand - 1st finger
        (9, 10),
        (10, 11),
        (11, 12),

        # left hand - 2nd finger
        (13, 14),
        (14, 15),
        (15, 16),

        # left hand - 3rd finger
        (17, 18),
        (18, 19),
        (19, 20),

        # left hand - 4th finger
        (21, 22),
        (22, 23),
        (23, 24),

        # left hand - 5th finger
        (25, 26),
        (26, 27),
        (27, 28),

        # right hand - wrist
        # (4, 29, 4),

        # right hand - palm
        (29, 30),
        (29, 34),
        (29, 38),
        (29, 42),
        (29, 46),

        # right hand - 1st finger
        (30, 31),
        (31, 32),
        (32, 33),

        # right hand - 2nd finger
        (34, 35),
        (35, 36),
        (36, 37),

        # right hand - 3rd finger
        (38, 39),
        (39, 40),
        (40, 41),

        # right hand - 4th finger
        (42, 43),
        (43, 44),
        (44, 45),

        # right hand - 5th finger
        (46, 47),
        (47, 48),
        (48, 49),
    )

def produce_length_direct(pose):
    pose_sliced = pose[..., :-1]
    pose_reshaped = pose_sliced.view(pose.shape[0], pose.shape[1], 50, 3)
    pose_list = pose_reshaped.split(1, dim=2)

    skeletons = getSkeletalModelStructure()

    length = []
    direct = []
    for skeleton in skeletons:
        result_length = torch.norm(pose_list[skeleton[0]] - pose_list[skeleton[1]], dim=3)
        length.append(result_length)
        epsilon = torch.finfo(result_length.dtype).tiny
        result_length = result_length + epsilon
        result_direct = torch.div((pose_list[skeleton[0]] - pose_list[skeleton[1]]).squeeze(), result_length)
        direct.append(result_direct)
    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=-1).reshape(lengths.shape[0],lengths.shape[1],47*3)

    return lengths, directs


def replace_nan_with_zero(tensor):
    tensor[torch.isnan(tensor)] = 0
    return tensor
class RegLoss(nn.Module):
    """
    Regression Loss
    """

    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()
        self.noises_loss = cfg["training"]["noises_loss"].lower()
        self.vel_loss = cfg["training"]["vel_loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.noises_loss == "l1":
            self.criterion_noise = nn.L1Loss()
        elif self.noises_loss == "mse":
            self.criterion_noise = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_noise = nn.MSELoss()

        if self.vel_loss == "l1":
            self.criterion_vel = nn.L1Loss()
        elif self.vel_loss == "mse":
            self.criterion_vel = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_vel = nn.MSELoss()

        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)


    # pylint: disable=arguments-differ
    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        preds_masked_length, preds_masked_direct = produce_length_direct(preds_masked)
        targets_masked_length, targets_masked_direct = produce_length_direct(targets_masked)

        preds_masked_length = preds_masked_length * loss_mask[:, :, :47]
        targets_masked_length = targets_masked_length * loss_mask[:, :, :47]
        preds_masked_direct = preds_masked_direct * loss_mask[:, :, :141]
        targets_masked_direct = targets_masked_direct * loss_mask[:, :, :141]


        # Calculate loss just over the masked predictions
        loss = self.criterion(preds_masked, targets_masked) + 0.1 * self.criterion_vel(preds_masked_direct, targets_masked_direct)
        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # standard xent loss
        self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                    reduction='sum')

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        # targets: indices with batch*seq_len
        targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)

        return loss