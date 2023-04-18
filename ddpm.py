import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = x1 + x2
        return out / 1.414


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        # flatten
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat)

        self.down_sample_1 = UnetDown(n_feat, n_feat)
        self.down_sample_2 = UnetDown(n_feat, 2 * n_feat)
        self.down_sample_3 = UnetDown(n_feat, 4 * n_feat)
        self.down_sample_4 = UnetDown(n_feat, 8 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.time_embed_1 = EmbedFC(1, 8 * n_feat)
        self.time_embed_2 = EmbedFC(1, 4 * n_feat)
        self.time_embed_3 = EmbedFC(1, 2 * n_feat)
        self.time_embed_4 = EmbedFC(1, 1 * n_feat)

        self.context_embed_1 = EmbedFC(n_classes, 8 * n_feat)
        self.context_embed_2 = EmbedFC(n_classes, 4 * n_feat)
        self.context_embed_3 = EmbedFC(n_classes, 2 * n_feat)
        self.context_embed_4 = EmbedFC(n_classes, 1 * n_feat)

        self.up_sample_0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up_sample_1 = UnetUp(16 * n_feat, n_feat)
        self.up_sample_2 = UnetUp(8 * n_feat, n_feat)
        self.up_sample_3 = UnetUp(4 * n_feat, n_feat)
        self.up_sample_4 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down_1 = self.down_sample_1(x)
        down_2 = self.down_sample_2(down_1)
        down_3 = self.down_sample_3(down_2)
        down_4 = self.down_sample_4(down_3)

        hidden_vec = self.to_vec(down_4)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
        c = c * context_mask

        # embed context, time step
        c_emb_1 = self.contextembed1(c).view(-1, self.n_feat * 8, 1, 1)
        c_emb_2 = self.contextembed2(c).view(-1, self.n_feat * 4, 1, 1)
        c_emb_3 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)
        c_emb_4 = self.contextembed2(c).view(-1, self.n_feat * 1, 1, 1)

        t_emb_1 = self.timeembed1(t).view(-1, self.n_feat * 8, 1, 1)
        t_emb_2 = self.timeembed2(t).view(-1, self.n_feat * 4, 1, 1)
        t_emb_3 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        t_emb_4 = self.timeembed2(t).view(-1, self.n_feat * 1, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up_sample_0(hidden_vec)

        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up_sample_1(c_emb_1 * up1 + t_emb_1, down_4)  # add and multiply embeddings
        up3 = self.up_sample_2(c_emb_2 * up2 + t_emb_2, down_3)
        up4 = self.up_sample_3(c_emb_3 * up3 + t_emb_3, down_2)
        up5 = self.up_sample_4(c_emb_4 * up4 + t_emb_4, down_1)

        out = self.out(torch.cat((up5, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        # generate a random step for each image in the batch
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)

        # sample 20 int from 1 to 101
        #torch.randint(1, 100 + 1, (20,))

        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        # torch.bernoulli(torch.zeros(3,3) + 0.2)
        # torch.bernoulli(torch.zeros(3,3) + 0.8)

        # return MSE between added noise, and our predicted noise

        # x_t is generated diffused image
        # c is ? predicted ?
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance (the bigger w, the more guidence)

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        # torch.randn(10, 10).shape
        # torch.randn(10, *(3, 3)).shape

        c_i = torch.arange(0, 10).to(device)  # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        # torch.arange(0, 10)
        # torch.arange(0, 10).repeat(2)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        torch.zeros_like(torch.arange(0, 10).repeat(2))


        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free


        x_i_store = []  # keep track of generated steps in case want to plot something
        print()

        list(range(10, 0, -1))

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # x = torch.tensor([[1, 2, 3], [1, 2, 3]])
            # x.repeat(2, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store



# c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
#
# tmp = torch.Tensor([0, 1,2,3,4])
# tmp = torch.Tensor([4,3,2,1,4])
#
# nn.functional.one_hot(tmp.long(), num_classes=5).type(torch.float)
#
#
# x = torch.rand((1, 3, 32, 32))
#
# model(x)


# a = ResidualConvBlock(in_channels=3, out_channels=4)
#
# b = UnetDown(in_channels=4, out_channels=8)
#
# c = UnetUp(in_channels=1, out_channels=3)
#
# d = EmbedFC(input_dim=1, emb_dim=3)
#
# x = torch.rand((1, 3, 32, 32))
#
# x2 = a(x)
#
# print(x2.shape)
#
# x3 = b(x2)
#
# print(x3.shape)
#
# x4 = d(x3)
#
# print(x4.shape)

# x.max()
# x.min()
# plt.imshow(x[0][0])
# plt.show()
