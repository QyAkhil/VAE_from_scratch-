import torch
import torch.nn as nn

class VariationalAutoencoders(nn.Module):

    def __init__(self, latent_dim):

        super(VariationalAutoencoders, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256*8*8, latent_dim)
        self.fc_logvar = nn.Linear(256*8*8, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256*8*8)

        self.decode = nn.Sequential(

            nn.Unflatten(1,(256,8,8)),

            nn.ConvTranspose2d(256,128,4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128,64,4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def reparametrize(self, mu, logvar):

        std = torch.exp(logvar/2)
        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def forward(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparametrize(mu, logvar)

        d = self.fc_decode(z)
        d = self.decode(d)

        return d, mu, logvar