from torchvision.utils import save_image
from ddpm.denoising import DenoiseDiffusion
import torch
from ddpm.utils import ImageLoader
from tqdm import tqdm
from config.config import logger

# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load UNet
from labml_nn.diffusion.ddpm.unet import UNet

eps_model = UNet(
    image_channels=3,
    n_channels=64,
    ch_mults=[1, 2, 2, 4],
    is_attn=[False, False, False, True],
).to(device)


class Trainer:
    def __init__(self, eps_model=eps_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DenoiseDiffusion(n_steps=1000, device=self.device, eps_model=eps_model)
        self.optimizer = torch.optim.Adam(self.diffusion.eps_model.parameters(), lr=2e-5)
        self.batch_size = 64
        self.seed = 3141
        self.g = torch.Generator(device=self.device).manual_seed(self.seed)
        self.data = ImageLoader().stage_data()

    def train(self, epochs: int = 1, checkpoint_path: str = "./pretrained_weights/"):
        # accelerate training computations
        torch.backends.cudnn.benchmark = True

        dataloader = torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

        for epoch in range(epochs):
            for step, batch in enumerate(tqdm(dataloader)):

                # get images from training batch (inefficient)
                train_batch = batch["image"]

                # move data to device
                train_batch = train_batch.to(self.device)

                # get MSE loss
                loss = self.diffusion.loss(train_batch)

                if step % 1000 == 0:
                    logger.info(
                        "[bold magenta]Epoch: {} Step: {} Loss: \
                        {}[/bold magenta]".format(
                            epoch, step, loss
                        )
                    )

                # collect gradients
                loss.backward()
                # perform optimization
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if step != 0 and step % 1000 == 0:
                    path = f"{checkpoint_path}" + f"model_{epoch}.pt"
                    torch.save(
                        {
                            "epoch": step,
                            "model_state_dict": self.diffusion.eps_model,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": loss,
                        },
                        path,
                    )

        path = f"{checkpoint_path}" + "model_final.pt"
        torch.save(
            {
                "epoch": step,
                "model_state_dict": self.diffusion.eps_model,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
