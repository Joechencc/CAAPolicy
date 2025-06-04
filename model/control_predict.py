import torch

from torch import nn
from timm.models.layers import trunc_normal_
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.tf_de_tgt_dim = self.cfg.future_frame_nums  #  T=8

        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.tf_de_tgt_dim, self.cfg.tf_de_dim) * .02
        )

        tf_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads
        )
        self.tf_decoder = nn.TransformerDecoder(
            tf_layer, num_layers=self.cfg.tf_de_layers
        )

        self.input_proj = nn.Linear(self.cfg.tf_de_dim, self.cfg.tf_de_dim)
        self.reg_head = nn.Sequential(
            nn.Linear(self.cfg.tf_de_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [acc, steer, reverse]
        )

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def decoder(self, encoder_out, tgt_embedding):
        encoder_out = encoder_out.transpose(0, 1)      # [S, B, D]
        tgt_embedding = tgt_embedding.transpose(0, 1)  # [T, B, D]
        output = self.tf_decoder(tgt=tgt_embedding, memory=encoder_out)
        return output.transpose(0, 1)  # [B, T, D]

    def forward(self, encoder_out):
        B = encoder_out.size(0)
        T = self.tf_de_tgt_dim

        dummy_input = torch.zeros(B, T, self.cfg.tf_de_dim, device=encoder_out.device)
        tgt_embedding = dummy_input + self.pos_embed[:, :T, :]
        tgt_embedding = self.pos_drop(tgt_embedding)

        decoder_out = self.decoder(encoder_out, tgt_embedding)
        control_out = self.reg_head(decoder_out)  # [B, T, 3]

        control_out = control_out.view(B, T * 3)  # flatten to [B, T*3]

        return control_out

    def predict(self, encoder_out):
        return self.forward(encoder_out)
