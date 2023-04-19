import os
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
        # the inputr dimension for one image is typically 1
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
        self.down_sample_3 = UnetDown(2 * n_feat, 4 * n_feat)
        self.down_sample_4 = UnetDown(4 * n_feat, 8 * n_feat)

        # 128/(2^4)=8  256/(2^4)=16  512/(2^4)=32

        self.avg_pool_size = int(128/(2**4))

        self.to_vec = nn.Sequential(nn.AvgPool2d(self.avg_pool_size), nn.GELU())

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
            nn.ConvTranspose2d(8 * n_feat, 8 * n_feat, self.avg_pool_size, self.avg_pool_size),
            nn.GroupNorm(8, 8 * n_feat),
            nn.ReLU(),
        )

        # nnn = nn.ConvTranspose2d(4, 4, 7, 7)
        # x = torch.rand(2, 4, 1, 1)
        #
        # nnn(x).shape

        self.up_sample_1 = UnetUp(16 * n_feat, 4 * n_feat)
        self.up_sample_2 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up_sample_3 = UnetUp(4 * n_feat, n_feat)
        self.up_sample_4 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, cc, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on
        # x = torch.concat([x,x,x,x], axis=2)
        # x = torch.concat([x,x,x,x], axis=3)

        x = self.init_conv(x)
        # x.shape
        # torch.Size([10, 64, 112, 112])

        down_1 = self.down_sample_1(x)
        # torch.Size([10, 64, 56, 56])

        down_2 = self.down_sample_2(down_1)
        # torch.Size([10, 128, 28, 28])

        down_3 = self.down_sample_3(down_2)
        # torch.Size([10, 256, 14, 14])

        down_4 = self.down_sample_4(down_3)
        # torch.Size([10, 512, 7, 7])

        hidden_vec = self.to_vec(down_4)
        # torch.Size([10, 512, 1, 1])

        # convert context to one hot embedding
        cc = nn.functional.one_hot(cc, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
        cc = cc * context_mask

        # embed context, time step
        c_emb_1 = self.context_embed_1(cc).view(-1, self.n_feat * 8, 1, 1)
        c_emb_2 = self.context_embed_2(cc).view(-1, self.n_feat * 4, 1, 1)
        c_emb_3 = self.context_embed_3(cc).view(-1, self.n_feat * 2, 1, 1)
        c_emb_4 = self.context_embed_4(cc).view(-1, self.n_feat * 1, 1, 1)

        t_emb_1 = self.time_embed_1(t).view(-1, self.n_feat * 8, 1, 1)
        t_emb_2 = self.time_embed_2(t).view(-1, self.n_feat * 4, 1, 1)
        t_emb_3 = self.time_embed_3(t).view(-1, self.n_feat * 2, 1, 1)
        t_emb_4 = self.time_embed_4(t).view(-1, self.n_feat * 1, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up_sample_0(hidden_vec)
        #breakpoint()
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

    def forward(self, x, cc):
        """
        this method is used in training, so samples t and noise randomly
        """

        # generate a random step for each image in the batch
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)

        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(cc) + self.drop_prob).to(self.device)

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, cc, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        c_i = torch.arange(0, 10).to(device)  # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

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


def train_mnist():

    # os.environ['CUDA_VISIABLE_DEVICES'] = ''
    # device_ids = [0,1]
    # model = torch.nn.DataParallel(


    # download data first
    # dataset = MNIST("D:/temp/teapearce/mnist", train=True, download=True)#, transform=tf)
    data_folder = "D:/temp/teapearce/mnist"
    save_dir = 'D:/temp/teapearce/diffusion_outputs10/'

    # hardcoding these here
    n_epoch = 1000
    batch_size = 60
    n_T = 500  # 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 32  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    # ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    ws_test = [0.0]  # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])  # mnist is already normalised 0 to 1

    dataset = MNIST(data_folder, train=True, download=True, transform=tf)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    ddpm = torch.nn.DataParallel(ddpm)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cc in pbar:
        #for x, cc in dataloader:
            optim.zero_grad()
            x = x.to(device)
            cc = cc.to(device)
            loss = ddpm(x, cc)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4 * n_classes
            for w_i, w in enumerate(ws_test):
                if isinstance(ddpm, torch.nn.DataParallel):
                    x_gen, x_gen_store = ddpm.module.sample(n_sample, (1, 128, 128), device, guide_w=w)
                else:
                    x_gen, x_gen_store = ddpm.sample(n_sample, (1, 128, 128), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample / n_classes)):
                        try:
                            idx = torch.squeeze((cc == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all * -1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep % 5 == 0 or ep == int(n_epoch - 1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, sharex=True, sharey=True,
                                            figsize=(8, 3))

                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample / n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(
                                    axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], cmap='gray',
                                                         vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots

                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval=200, blit=False, repeat=True,
                                        frames=x_gen_store.shape[0])
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_mnist()
