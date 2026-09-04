
from dataclasses import dataclass, field
from typing import List


import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


@dataclass
class AEConfig:
    """configurations of AE models"""
    dim_emb: int = 512 # 'dimension of word embedding
    dim_h: int = 1024 # dimension of hidden state per layer
    dropout: float = 0.5
    nlayers: int = 1 # number of LSTM layers
    dim_z: int = 128 # dimension of latent variable z
    lr: float = 1e-4 # learning rate
    vocab_size:int = 950
    pad_token:int = 1
    lambda_kl:float = 0.1


class AEBase(nn.Module):
    """base class for all the AE based models"""

    def __init__(self, config:AEConfig, initrange=0.1):
        super().__init__()
        # self.vocab = vocab
        # self.args = args
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, self.config.dim_emb)
        self.proj = nn.Linear(self.config.dim_h, config.vocab_size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

    def get_num_params(self, non_embedding=True):
        """get the nubmer of parameters"""
        return sum(p.numel() for p in self.parameters()) / 1e6 # print in terms of millions

    def reparameterize(self, mu, logvar):
        """the reparameterization trick"""

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def log_prob(self, z, mu, logvar):
        """recontruction loss"""

        var = torch.exp(logvar)
        logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
        return logp.sum(dim=1)
    
    def loss_kl(self, mu, logvar):

        """kl divergence loss"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class DAE(AEBase):
    """Denoising Auto-Encoder"""

    def __init__(self, config:AEConfig):
        super().__init__(config)
        self.drop = nn.Dropout(config.dropout)

        # produces 2 hidden vectors, one for the mu and the other for the logvar (sigma)
        self.encoder = nn.LSTM(config.dim_emb, config.dim_h, config.nlayers,
            dropout=config.dropout if config.nlayers > 1 else 0, bidirectional=True)
        
        self.generator = nn.LSTM(config.dim_emb, config.dim_h, config.nlayers,
            dropout=config.dropout if config.nlayers > 1 else 0)
        self.h2mu = nn.Linear(config.dim_h*2, config.dim_z)
        self.h2logvar = nn.Linear(config.dim_h*2, config.dim_z)
        self.z2emb = nn.Linear(config.dim_z, config.dim_emb)
        self.opt = torch.optim.Adam(self.parameters(), lr=config.lr, betas=(0.5, 0.999))

    def flatten(self):
        self.encoder.flatten_parameters()
        self.generator.flatten_parameters()

    def encode(self, input):

        input = self.drop(self.embed(input))
        _, (h, _) = self.encoder(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        

        input = self.drop(self.embed(input)) + self.z2emb(z) #.unsqueeze(1)
        output, hidden = self.generator(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, is_train=False):
        """
        input: (T, B)
        """
        _input = self.noisy(input) if is_train else input # TO DO, finish implementing the moisy method

        mu, logvar = self.encode(_input)
        z = self.reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)

        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):

        # ipdb.set_trace()

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.config.pad_token, reduction='none').view(targets.size())
        
        # ipdb.set_trace()
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)

        if is_train:
            return {'rec': self.loss_rec(logits, targets).mean()}
        else:
            return {'rec': self.loss_rec(logits, targets)}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is

    def noisy(self, input):
        # if shuffle_dist > 0:
        #     x = word_shuffle(vocab, x, shuffle_dist)
        # if drop_prob > 0:
        #     x = word_drop(vocab, x, drop_prob)
        # if blank_prob > 0:
        #     x = word_blank(vocab, x, blank_prob)
        # if sub_prob > 0:
        #     x = word_substitute(vocab, x, sub_prob)
        return input


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, config):
        super().__init__(config)

    def loss(self, losses):
        return losses['rec'] + self.config.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)

        if is_train:
            return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': self.loss_kl(mu, logvar)}
        else:
            return {'rec': self.loss_rec(logits, targets),
                'kl': self.loss_kl(mu, logvar)}
        
    