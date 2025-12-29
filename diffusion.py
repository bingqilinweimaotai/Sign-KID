# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from typing import List
from collections import namedtuple
from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn
from decoders import TransformerDecoder


__all__ = ["Diffusion"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def getSkeletalModelStructure():
    return (
        # head
        (1, 0),

        (1, 1),     # 中心

        (1, 2),

        # left arm
        (2, 3),

        (4, 4),      # 舍弃

        (1, 5),

        (5, 6),

        (7, 7),     # 舍弃

        (6, 8),

        (8, 9),

        (9, 10),

        (10, 11),

        (11, 12),

        (8, 13),

        (13, 14),

        (14, 15),

        (15, 16),

        (8, 17),

        (17, 18),

        (18, 19),

        (19, 20),

        (8, 21),

        (21, 22),

        (22, 23),

        (23, 24),

        (8, 25),

        (25, 26),

        (26, 27),

        (27, 28),

        (3, 29),

        (29, 30),

        (30, 31),

        (31, 32),

        (32, 33),

        (29, 34),

        (34, 35),

        (35, 36),

        (36, 37),

        (29, 38),

        (38, 39),

        (39, 40),

        (40, 41),

        (29, 42),

        (42, 43),

        (43, 44),

        (44, 45),

        (29, 46),

        (46, 47),

        (47, 48),

        (48, 49),

    )


def produce_data(trg):
    trg_information, counter = torch.split(trg.clone(), [150, 1], dim=-1)
    trg_reshaped = trg_information.view(trg.shape[0], trg.shape[1], 50, 3)
    trg_list = trg_reshaped.split(1, dim=2)

    skeletons = getSkeletalModelStructure()

    trg_reshaped_list = []

    for skeleton in skeletons:
        if skeleton[0] == skeleton[1]:
            trg_reshaped_list.append(torch.cat((trg_list[skeleton[0]].squeeze(),torch.zeros(trg.shape[0], trg.shape[1], 4).to("cuda:0")),dim=2))
        else:
            result_length = torch.norm(trg_list[skeleton[0]] - trg_list[skeleton[1]], dim=3)
            epsilon = torch.finfo(result_length.dtype).tiny
            length_epsilon = result_length + epsilon
            result_direct = torch.div((trg_list[skeleton[0]] - trg_list[skeleton[1]]).squeeze(), length_epsilon)
            trg_reshaped_list.append(torch.cat((torch.cat((trg_list[skeleton[1]].squeeze(), result_length), dim=2), result_direct), dim=2))
    trg_super = torch.stack(trg_reshaped_list, dim=-1).reshape(trg.shape[0],trg.shape[1],50*7)
    return trg_super

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    """
    Implement D3DP
    """

    def __init__(self, args, trg_vocab, sampling_timesteps=1):
        super().__init__()

        # Build diffusion
        timesteps = args["diffusion"].get('timesteps', 1000)
        # Multi-step generation
        sampling_timesteps = sampling_timesteps
        # Noise enhancement parameters
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args["diffusion"].get('scale', 1.0)
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.trg_embed = nn.Linear(350, args["diffusion"].get('cs', 512))
        self.pose_estimator = TransformerDecoder(num_layers=args["diffusion"].get('dep', 4),
                                                 num_heads=4,
                                                 hidden_size=args["diffusion"].get('cs', 512),
                                                 ff_size=2048,
                                                 dropout=args["decoder"].get('dropout', 0.1),
                                                 emb_dropout=args["decoder"]["embeddings"].get('dropout', 0.1),
                                                 vocab_size=len(trg_vocab),
                                                 freeze=False,
                                                 trg_size=args.get('trg_size', 150)+1,
                                                 decoder_trg_trg_=True)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, encoder_output, t,src_mask,trg_mask):
        x_t = produce_data(x)
        x_t = x_t / self.scale

        pred_pose = self.pose_estimator(encoder_output=encoder_output,
                                        trg_embed=x_t,
                                        src_mask=src_mask,
                                        trg_mask=trg_mask,
                                        t=t)

        x_start = pred_pose
        x_start = x_start * self.scale
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, encoder_output, input_3d, src_mask, trg_mask):
        batch = encoder_output.shape[0]
        shape = (batch, input_3d.shape[1], 151)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        img = torch.randn(shape, device='cuda:0')

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        preds_all=[]
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device='cuda:0', dtype=torch.long)

            preds = self.model_predictions(x=img, encoder_output=encoder_output, t=time_cond,src_mask=src_mask, trg_mask=trg_mask)
            pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return preds_all

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, encoder_output, input_3d, src_mask, trg_mask,is_train):

        # Prepare Proposals.
        if not is_train:
            results = self.ddim_sample(encoder_output=encoder_output, input_3d=input_3d, src_mask=src_mask, trg_mask=trg_mask)
            return results[self.sampling_timesteps-1]

        if is_train:
            x_poses, noises, t = self.prepare_targets(input_3d)
            x_poses = x_poses.float()
            x_poses = produce_data(x_poses)
            t = t.squeeze(-1)
            pred_pose = self.pose_estimator(encoder_output=encoder_output,
                                            trg_embed=x_poses,
                                            src_mask=src_mask,
                                            trg_mask=trg_mask,
                                            t=t)
            return pred_pose

    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(pose_3d.shape[0],151, device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = x / self.scale


        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)