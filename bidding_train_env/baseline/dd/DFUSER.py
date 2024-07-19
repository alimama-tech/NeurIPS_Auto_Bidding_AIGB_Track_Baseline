from torch.optim import Adam
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


def extract(a, t, x_shape: list):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, 1, 1)


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim: int):
    x[:, :conditions.shape[0], action_dim:] = conditions

    return x


class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ, masks: torch.Tensor):
        loss = self._loss(pred, targ)
        if masks is not None:
            loss = loss * masks[:, :, None].float()
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}


class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'state_l2': WeightedStateL2,
}


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            dim=128,
            dim_mults=(1, 2, 4),
            returns_condition=True,
            condition_dropout=0.1,
            calc_energy=False,
            kernel_size=5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        self.returns_mlp = nn.Sequential(
            nn.Linear(1, dim),
            act_fn,
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        embed_dim = 2 * dim
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon,
                                                kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon,
                                                kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon,
                                      kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size,
                                      mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns: torch.Tensor = torch.ones(1, 1), use_dropout: bool = True,
                force_dropout: bool = False):

        x = torch.permute(x, (0, 2, 1))

        # print(returns.shape)
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask_index = torch.randperm(returns_embed.shape[0])[
                             :int(returns_embed.shape[0] * self.condition_dropout)]
                mask = torch.ones_like(returns_embed, device=returns_embed.device)
                mask[mask_index] = 0
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for models in self.downs:
            resnet = models[0]
            resnet2 = models[1]
            downsample = models[2]
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for models in self.ups:
            resnet = models[0]
            resnet2 = models[1]
            upsample = models[2]
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = torch.permute(x, (0, 2, 1))

        return x


class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 clip_denoised=False, predict_epsilon=True, hidden_dim=256,
                 loss_discount=1.0, returns_condition=False,
                 condition_guidance_w=0.1):
        super().__init__()

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.inv_model = nn.Sequential(
            nn.Linear(4 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount):

        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.matmul(discounts[:, None], dim_weights[None, :])

        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns: torch.Tensor = torch.ones(1, 1)):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself

            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, cond, t, returns: torch.Tensor = torch.ones(1, 1)):
        with torch.no_grad():
            b, _, _ = x.shape
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
            noise = 0.5 * torch.randn_like(x, device=x.device)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, 1, 1)
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape, cond, returns: torch.Tensor = torch.ones(1, 1)):
        with torch.no_grad():
            torch.random.manual_seed(1019)
            batch_size = shape[0]
            x = 0.5 * torch.randn(shape[0], shape[1], shape[2], device=cond.device)

            x = apply_conditioning(x, cond, 0)

            for i in range(self.n_timesteps - 1, -1, -1):
                timesteps = torch.ones(batch_size,
                                       device=cond.device) * i
                x = self.p_sample(x, cond, timesteps, returns)

                x = apply_conditioning(x, cond, 0)

            return x

    #  @torch.no_grad()
    def conditional_sample(self, cond, returns: torch.Tensor = torch.ones(1, 1), horizon: int = 48):
        with torch.no_grad():
            batch_size = 1
            horizon = self.horizon
            shape = torch.tensor([batch_size, horizon, self.observation_dim])

            return self.p_sample_loop(shape, cond, returns)

    def forward(self, cond, returns):
        return self.conditional_sample(cond=cond, returns=returns)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(t.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(t.device)
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None, masks=None):
        noise = torch.randn_like(x_start, device=x_start.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        t = t.to(x_noisy.device)
        x_recon = self.model(x_noisy, cond, t, returns)

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, masks)
        else:
            loss, info = self.loss_fn(x_recon, x_start, masks)

        return loss, info

    def loss(self, x, cond, returns, masks):

        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns, masks)
        # Calculating inv loss
        x_t = x[:, :-1, self.action_dim:]
        a_t = x[:, :-1, :self.action_dim]
        x_t_1 = x[:, 1:, self.action_dim:]
        x_t_2 = torch.cat(
            [torch.zeros(x.shape[0], 1, x.shape[-1] - self.action_dim, device=x.device), x[:, :-2, self.action_dim:]],
            dim=1)
        x_t_3 = torch.cat(
            [torch.zeros(x.shape[0], 2, x.shape[-1] - self.action_dim, device=x.device), x[:, :-3, self.action_dim:]],
            dim=1)
        x_comb_t = torch.cat([x_t_2, x_t_3, x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 4 * self.observation_dim)
        masks_flat = masks[:, :-1].reshape(-1)
        x_comb_t = x_comb_t[masks_flat]
        a_t = a_t.reshape(-1, self.action_dim)
        a_t = a_t[masks_flat]
        pred_a_t = self.inv_model(x_comb_t)
        inv_loss = F.mse_loss(pred_a_t, a_t)
        loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info, (diffuse_loss, inv_loss)


class DFUSER(nn.Module):
    def __init__(self, dim_obs=16, dim_actions=1, gamma=1, tau=0.01, lr=1e-4,
                 network_random_seed=200,
                 ACTION_MAX=10, ACTION_MIN=0,
                 step_len=48, n_timesteps=10):

        super().__init__()

        self.n_timestamps = n_timesteps
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.ACTION_MAX = ACTION_MAX
        self.ACTION_MIN = ACTION_MIN
        self.network_random_seed = network_random_seed
        self.step_len = step_len

        model = TemporalUnet(
            horizon=step_len,
            transition_dim=dim_obs,
            cond_dim=dim_actions,
            returns_condition=True,
            dim=128,
            condition_dropout=0.25,
            calc_energy=False
        )

        self.diffuser = GaussianInvDynDiffusion(
            model=model,
            horizon=step_len,
            observation_dim=dim_obs,
            action_dim=dim_actions,
            clip_denoised=True,
            predict_epsilon=True,
            hidden_dim=256,
            n_timesteps=n_timesteps,
            loss_discount=1,
            returns_condition=True,
            condition_guidance_w=1.2
        )

        self.step = 0

        torch.random.manual_seed(network_random_seed)

        self.num_of_episodes = 0

        self.GAMMA = gamma
        self.tau = tau
        self.num_of_steps = 0
        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.diffuser.cuda()

        self.diffuser_lr = lr

        self.diffuserModel_optimizer = torch.optim.Adam(self.diffuser.model.parameters(), lr=lr)
        self.invModel_optimizer = torch.optim.Adam(self.diffuser.inv_model.parameters(), lr=lr)

    def toCuda(self):
        self.diffuser.cuda()

    def trainStep(self, states, actions, returns, masks):
        self.diffuser.train()
        if self.use_cuda:
            self.diffuser.cuda()
            states = states.cuda()
            actions = actions.cuda()
            returns = returns.cuda()
            masks = masks.cuda()

        x = torch.cat([actions, states], dim=-1)
        cond = torch.ones_like(states[:, 0], device=states.device)[:, None, :]
        loss, infos, (diffuse_loss, inv_loss) = self.diffuser.loss(x, cond, returns=returns, masks=masks)
        inv_loss.backward()
        self.invModel_optimizer.step()
        self.invModel_optimizer.zero_grad()

        diffuse_loss.backward()
        self.diffuserModel_optimizer.step()
        self.diffuserModel_optimizer.zero_grad()

        return loss, (diffuse_loss, inv_loss)

    def forward(self, x: torch.Tensor):
        if len(list(x.shape)) < 2:
            x = torch.reshape(x, [48, self.num_of_states + 1])
        else:
            x = x[0][0]
        cur_time = int(x[0][-1].item())
        cur_time = cur_time + 1
        states = x[:cur_time]
        states = states[:, :-1]
        conditions = states
        returns = torch.tensor([[1.0]], device=x.device)
        x_0 = self.diffuser(cond=conditions, returns=returns)

        states = x_0[0, :cur_time + 1]
        states_next = states[None, -1]
        if cur_time > 1:
            states_curt1 = conditions[-2].float()[None, :]
        else:
            states_curt1 = torch.zeros_like(states_next, device=states_next.device)
        if cur_time > 2:
            states_curt2 = conditions[-3].float()[None, :]
        else:
            states_curt2 = torch.zeros_like(states_next, device=states_next.device)
        states_comb = torch.hstack([states_curt1, states_curt2, conditions[-1].float()[None, :], states_next])
        actions = self.diffuser.inv_model(states_comb)
        actions = actions.detach().cpu()[0]  # .cpu().data.numpy()
        return actions

    def save_net(self, save_path, epi):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.diffuser.state_dict(), f'{save_path}/diffuser.pt')

    def save_model(self, save_path, epi):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        model_temp = self.cpu()
        jit_model = torch.jit.script(model_temp)
        torch.jit.save(jit_model, f'{save_path}/diffuser_{epi}.pth')

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0'):
        self.diffuser.load_state_dict(torch.load(load_path, map_location='cpu'))
        self.optimizer = Adam(self.diffuser.parameters(), lr=self.diffuser_lr)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.diffuser.cuda()
