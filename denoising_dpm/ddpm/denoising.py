import torch
import torch.nn.functional as F
import tqdm
from labml_nn.diffusion.ddpm.unet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DenoiseDiffusion:
    def __init__(self, n_steps, device: torch.device):
        super().__init__()
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        self.n_samples = 10
        self.image_channels = 3
        self.image_size = 32
        self.device = device
        self.eps_model = UNet(
            image_channels = 3,
            n_channels = 64,
            ch_mults = [1, 2, 2, 4],
            is_attn = [False, False, False, True],
        ).to(device)

    def extract(self, consts, t):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

    def q_xt_x0(self, x0, t):
        """Get $q(x_t|x_0)$ distribution"""
        # 1. Gather αt and compute (sqrt(αt_bar) * x0) for the mean
        mean = self.extract(self.alpha_bar, t) ** 0.5 * x0
        # 2. Get variance σ
        var = 1. - self.extract(self.alpha_bar, t)
        # 3. Return mean µ and variance σ
        return mean, var

    def q_sample(self, x0, t, eps=None):
        """Sample from $q(x_t|x_0)$. Equation #4 from paper."""
        # 1. Sample epsilon from Gaussian distribution
        if eps is None:
            eps = torch.randn_like(x0)
        # 2. Get mean and variance from q_xt_x0
        mean, var = self.q_xt_x0(x0, t)
        # 3. Return sample from q_xt_x0
        return mean + (var ** 0.5) * eps

    def loss(self, x0, noise=None):
        """
        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - {\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$

        """
        # 1. Get batch size
        batch_size = x0.shape[0]
        # 2. Get timestep tensor
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        # 3. Sample noise from Gaussian distribution
        if noise is None:
            noise = torch.randn_like(x0)
        # 4. Sample from $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # 5. Get $\epsilon_\theta$
        eps_theta = self.eps_model(xt, t)
        # 6. Compute loss between sampled and predicted noise
        return F.mse_loss(noise, eps_theta)
    
    def p_sample(self, xt, t, eps_theta):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        """
        # 1. Compute alphas
        alpha_bar = self.alpha_bar
        alpha = self.alpha
        # 2. Extract alpha and alpha_bar at timestep t
        alpha_bar = self.extract(alpha_bar, t)
        alpha = self.extract(alpha, t)
        # 3. Get predicted noise at timestep t
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        # 4. Compute posterior mean
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # 5. Compute posterior variance
        var = self.extract(self.sigma2, t)
        # 6. Sample Gaussian noise
        eps = torch.randn(xt.shape, device=xt.device)
        # 7. Return $x_{t-1}$
        return mean + (var ** 0.5) * eps

    @torch.no_grad()
    def sampling(self):
        """
        Algorithm 2 from paper.
        """
        # 1: sample x_T ~ N(0, I)
        xt = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size], device=self.device)

        # 2: for t = T-1, ..., 0 do
        for t_inv in tqdm(range(self.n_steps)):
            t_ = self.n_steps - t_inv
            t = xt.new_full((self.n_samples,), t_, dtype=torch.long)
        
            eps_theta = self.eps_model(xt, t)

            xt = self.p_sample(xt, t, eps_theta)
        


            

        

