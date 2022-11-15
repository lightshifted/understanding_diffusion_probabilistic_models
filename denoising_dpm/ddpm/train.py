from torchvision.utils import save_image
from ddpm.denoising import DenoisingDiffusion
import torch
from ddpm.utils import DataLoader
import tqdm


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DenoisingDiffusion(n_steps=1000, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=2e-5)
        self.n_epochs = 5
        self.batch_size = 64
        self.seed = 3141
        self.g = torch.Generator(device=self.device).manual_seed(self.seed)
        self.data = DataLoader().stage_data()

    def train(self):
        # accelerate training computations
        torch.backends.cudnn.benchmark = True
        
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        for epoch in range(self.epochs):
            for step, batch in enumerate(tqdm(dataloader)):

                # get images from training batch (inefficient)
                train_batch = batch['image']

                # move data to device
                train_batch = train_batch.to(self.device)

                # get MSE loss
                loss = self.diffusion.loss(train_batch)

                if step % 1000 == 0:
                    print("[bold magenta]Loss:[/bold magenta] %.4f" % (loss.item()))

                # collect gradients
                loss.backward()
                # perform optimization
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if step != 0 and step % 1000 == 0:
                    path = f'./pretrained_weights/model_{epoch}.pt'
                    torch.save({
                        'epoch': step,
                        'model_state_dict': self.diffusion.eps_model,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                    }, path)

        torch.save({
            'epoch': step,
            'model_state_dict': self.diffusion.eps_model,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
