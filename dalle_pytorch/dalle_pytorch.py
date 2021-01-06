from math import log2
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from x_transformers import Encoder, Decoder

# helpers

def exists(val):
    return val is not None

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

# classes

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim = 512,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, num_tokens, 1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )

        self.codebook = nn.Embedding(num_tokens, dim)

    def forward(
        self,
        img,
        return_recon_loss = False
    ):
        logits = self.encoder(img)
        soft_one_hot = F.gumbel_softmax(logits, tau = 1.)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_recon_loss:
            return out

        loss = F.mse_loss(img, out)
        return loss

# main classes

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        num_text_tokens = 10000,
        num_visual_tokens = 512,
        text_enc_depth = 6,
        visual_enc_depth = 6,
        text_seq_len = 256,
        visual_seq_len = 1024,
        text_heads = 8,
        visual_heads = 8
    ):
        super().__init__()
        self.scale = dim ** -0.5
        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.visual_emb = nn.Embedding(num_visual_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len, dim)
        self.visual_pos_emb = nn.Embedding(visual_seq_len, dim)

        self.text_transformer = Encoder(dim = dim, depth = text_enc_depth, heads = text_heads)
        self.visual_transformer = Encoder(dim = dim, depth = visual_enc_depth, heads = visual_heads)

    def forward(
        self,
        text,
        image,
        text_mask = None,
        return_similarities = False
    ):
        b, device = text.shape[0], text.device

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_emb = self.visual_emb(image)
        image_emb += self.visual_pos_emb(torch.arange(image.shape[1], device = device))

        enc_text = self.text_transformer(text_emb)
        enc_image = self.visual_transformer(image_emb)

        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim = 1)
        else:
            text_latents = enc_text.mean(dim = 1)

        image_latents = enc_image.mean(dim = 1)

        sim = einsum('i d, j d -> i j', text_latents, image_latents) * self.scale

        if return_similarities:
            return sim

        labels = torch.arange(b, device = device)
        loss = F.cross_entropy(sim, labels)
        return loss


class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_text_tokens = 10000,
        num_image_tokens = 512,
        text_seq_len = 256,
        image_seq_len = 1024,
        depth = 6, # should be 64
        heads = 8
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len, dim)
        self.image_pos_emb = nn.Embedding(image_seq_len, dim)

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.image_seq_len = image_seq_len
        self.total_tokens = num_text_tokens + num_image_tokens + 1 # extra for EOS

        self.transformer = Decoder(dim = dim, depth = depth, heads = heads)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

    def forward(
        self,
        text,
        image,
        mask = None,
        return_loss = False
    ):
        device = text.device

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_emb = self.image_emb(image)
        image_emb += self.image_pos_emb(torch.arange(image.shape[1], device = device))

        tokens = torch.cat((text_emb, image_emb), dim = 1)

        if exists(mask):
            mask = F.pad(mask, (0, self.image_seq_len), value = True)

        out = self.transformer(tokens, mask = mask)
        out = self.to_logits(out)

        if not return_loss:
            return out

        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text, offsetted_image), dim = 1)
        labels = F.pad(labels, (0, 1), value = (self.total_tokens - 1)) # last token predicts EOS
        loss = F.cross_entropy(out.transpose(1, 2), labels[:, 1:])
        return loss