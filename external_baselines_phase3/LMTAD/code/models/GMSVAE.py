
"""
References:
    this model is based on the GMSVAE repository found at this location: This code is based on the following github: https://github.com/liuyiding1993/ICDE2020_GMVSAE/blob/master/preprocess/preprocess.py

    We reimplemented the code in pytorch
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GMSVAEConfig:
    """configurations of GMSVAE models"""
    dim_emb: int = 512 # 'dimension of word embedding
    dim_h: int = 1024 # dimension of hidden state per layer
    dropout: float = 0.2
    nlayers: int = 1 # number of LSTM layers
    dim_z: int = 128 # dimension of latent variable z
    components: int = 10
    lr: float = 1e-4 # learning rate
    vocab_size:int = 513
    pad_token:int = 1
    lambda_kl:float = 0.1

class Seq2Seq(nn.Module):
    """RNN Seq2Seq model"""
    def __init__(self, cell, embeddings, role):
        super(Seq2Seq, self).__init__()
        self.cell = cell
        self.embeddings = embeddings
        if role not in ['encoder', 'decoder']:
            raise ValueError("The role must be 'encoder' or 'decoder'.")
        self.role = role

    def forward(self, tokens, seq_lengths, initial_state=None):
        """forward method"""

        token_embeds = self.embeddings(tokens)

        # if self.role == 'decoder':
        #     ipdb.set_trace()

        packed = nn.utils.rnn.pack_padded_sequence(token_embeds, seq_lengths, batch_first=True, enforce_sorted=False)

        if initial_state is None:
            outputs, final_state = self.cell(packed)
        else:
            outputs, final_state = self.cell(packed, initial_state)
        if self.role == 'encoder':
            return final_state
        else:  # self.role == 'decoder'
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            return outputs

class LatentGaussianMixture(nn.Module):
    """guassian miture model"""
    def __init__(self, config:GMSVAEConfig):
        super(LatentGaussianMixture, self).__init__()
        self.config = config

        # Initialize mu_c either with pretrained values or randomly
        # if args.pretrain_dir:
        #     mu_c_path = f"{args.pretrain_dir}/{args.model}_{args.token_dim}_{args.rnn_dim}_{args.cluster_num}/init_mu_c.npz"
        #     mu_c = np.load(mu_c_path)['arr_0']
        #     self.mu_c = nn.Parameter(torch.tensor(mu_c, dtype=torch.float32))
        # else:

        self.mu_c = nn.Parameter(torch.randn(config.components, config.dim_h))
        self.log_sigma_sq_c = nn.Parameter(torch.zeros(config.components, config.dim_h)) #, requires_grad=False) why did they have that?

        # Fully connected layers for mu_z and log_sigma_sq_z
        self.fc_mu_z = nn.Linear(config.dim_h, config.dim_h)
        self.fc_sigma_z = nn.Linear(config.dim_h, config.dim_h)

    def forward(self, input, return_loss=False):

        """sample and compute the guassian loss and uniform loss?
            input: input: (B, T) -> output of the encoder
        
        """

        B, T = input.size()
        mu_z = self.fc_mu_z(input)
        log_sigma_sq_z = self.fc_sigma_z(input)

        eps_z = torch.randn_like(log_sigma_sq_z)
        z = mu_z + torch.exp(0.5 * log_sigma_sq_z) * eps_z  # Reparameterization trick

        stack_z = z.unsqueeze(1).expand(-1, self.config.components, -1)
        stack_mu_c = self.mu_c.expand(B, -1, -1)
        stack_mu_z = mu_z.unsqueeze(1).expand(-1, self.config.components, -1)
        stack_log_sigma_sq_z = log_sigma_sq_z.unsqueeze(1).expand(-1, self.config.components, -1)
        stack_log_sigma_sq_c = self.log_sigma_sq_c.expand(B, -1, -1)

        pi_post_logits = -torch.sum((stack_z - stack_mu_c)**2 / torch.exp(stack_log_sigma_sq_c), dim=-1)
        pi_post = F.softmax(pi_post_logits, dim=-1) + 1e-10

        if not return_loss:
            return z
        else:
            batch_gaussian_loss = 0.5 * torch.sum(
                pi_post * torch.mean(
                    stack_log_sigma_sq_c +
                    torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c) +
                    (stack_mu_z - stack_mu_c)**2 / torch.exp(stack_log_sigma_sq_c),
                    dim=-1
                ), dim=-1) - 0.5 * torch.mean(1 + log_sigma_sq_z, dim=-1)

            batch_uniform_loss = torch.mean(torch.mean(pi_post, dim=0) * torch.log(torch.mean(pi_post, dim=0)))

        return z, [batch_gaussian_loss, batch_uniform_loss]


class GMSVAE(nn.Module):
    """the GMSVAE model"""
    def __init__(self, config:GMSVAEConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.dim_emb)

        # Encoder
        encoder_cell = nn.GRU(self.config.dim_emb, config.dim_h, batch_first=True)
        self.encoder = Seq2Seq(encoder_cell, self.embeddings, 'encoder')

        # Latent Space
        self.latent_space = LatentGaussianMixture(self.config)

        # Decoder
        decoder_cell = nn.GRU(self.config.dim_emb, self.config.dim_h, batch_first=True)
        self.decoder = Seq2Seq(decoder_cell, self.embeddings, 'decoder')

        # Output layer
        # self.out_w = nn.Parameter(torch.randn(self.config.vocab_size, self.config.dim_emb))
        # self.out_b = nn.Parameter(torch.randn(self.config.vocab_size))

        self.proj = nn.Linear(self.config.dim_h, self.config.vocab_size)

        self.opt = torch.optim.Adam(self.parameters(), lr=config.lr, betas=(0.5, 0.999))

    def get_num_params(self, non_embedding=True):
        """get the nubmer of parameters"""
        return sum(p.numel() for p in self.parameters()) / 1e6 # print in terms of millions
        
    def forward(self, inputs, masks):
        """the forward method"""
        # tokens, masks, seq_lengths = inputs
        
        seq_lengths = masks.sum(-1).to(torch.int32).cpu()
        B, T = inputs.size()
        
        # Encoder
        encoder_final_state = self.encoder(inputs, seq_lengths)
        
        # Latent Space
        z, latent_losses = self.latent_space(encoder_final_state[-1], return_loss=True)
        
        # Decoder
        batch_zeros = torch.zeros((B, 1), dtype=torch.int32).to(inputs.device)
        # tokens_shifted = torch.cat([batch_zeros, inputs[:, :-1]], dim=1)
        tokens_with_zeros = torch.cat([batch_zeros, inputs], dim=1) # TODO, remove these lines becasue they don't make sense
        targets = torch.cat([inputs, batch_zeros], dim=1)
        masks = torch.cat([masks, batch_zeros], dim=1)

        seq_lengths = seq_lengths + 1 # to remove
        outputs = self.decoder(tokens_with_zeros, seq_lengths, initial_state=z.unsqueeze(0))

        # Output calculation (adapt as necessary for your specific model)
        logits = self.proj(outputs)

        # Loss calculation (this should be adapted as per your loss function requirements)
        # Losses like reconstruction loss, gaussian loss, etc., should be calculated here or outside the model based on your design.

        return logits, latent_losses, targets, z, masks
    
    def compute_losses(self, logits, targets, masks, latent_losses):
        """compute the losses"""
        

        # Assuming targets are provided as class indices for cross-entropy loss
        # Flatten outputs and targets for loss calculation
        outputs_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute reconstruction loss (e.g., cross-entropy)
        rec_loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')
        
        rec_loss = rec_loss.view_as(targets) * masks

        rec_loss = rec_loss.sum(-1) / masks.sum(-1)

        # Compute Gaussian mixture loss and any other components of latent_losses
        # Example: gaussian_loss = latent_losses[0] (adjust as per your implementation)
        gaussian_loss, uniform_loss = latent_losses
        
        # Combine losses (modify coefficients as needed)
        # total_loss = rec_loss + gaussian_loss + uniform_loss

        # ipdb.set_trace()

        return rec_loss, gaussian_loss, uniform_loss
    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()
    
    def loss(self, losses):
        return losses['rec'] + losses['loss_gauss'] + losses['loss_uniform']
    def autoenc(self, inputs, masks, is_train=False):
        
        logits, latent_losses, targets, z, masks_ = self(inputs, masks)
        rec_loss, gaussian_loss, uniform_loss = self.compute_losses(logits, targets, masks_, latent_losses)

        # ipdb.set_trace()
        if is_train:
            return {'rec': rec_loss.mean(),
                'loss_gauss': gaussian_loss.mean(), "loss_uniform": uniform_loss}
        else:
            return {'rec': rec_loss,
                'loss_gauss': gaussian_loss, "loss_uniform": uniform_loss}
