import torch

from torch import nn
from timm.models.layers import trunc_normal_
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1

        self.embedding = nn.Embedding(self.cfg.token_nums, self.cfg.tf_de_dim-45)
        self.input_proj = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 45))
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.tf_de_tgt_dim - 1, self.cfg.tf_de_dim) * .02)

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def create_mask(self, tgt):
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).cuda()
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = (tgt == self.pad_idx)
        return tgt_mask, tgt_padding_mask

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask):
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        pred_controls = self.tf_decoder(tgt=tgt_embedding,
                                        memory=encoder_out,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)
        pred_controls = pred_controls.transpose(0, 1)
        return pred_controls

    # original forward
    # def forward(self, encoder_out, tgt):
    #     tgt = tgt[:, :-1]
    #     tgt_mask, tgt_padding_mask = self.create_mask(tgt)

    #     tgt_embedding = self.embedding(tgt)
    #     tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)

    #     pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
    #     pred_controls = self.output(pred_controls)
    #     return pred_controls

    def forward(self, encoder_out, batch):
        B = batch['gt_control'].shape[0]

        # ---------------------------------------------------
        # Use first 14 tokens (start, 12 control, end)
        # ---------------------------------------------------
        gt_control = batch['gt_control'][:, :-1]           # [B, 14]
        control_tokens = gt_control.unsqueeze(-1)          # [B, 14, 1]

        # ---------------------------------------------------
        # Expand step-wise inputs: 4 → 12 → insert start & end
        # ---------------------------------------------------
        def expand_and_pad(x):
            x_rep = x.repeat_interleave(3, dim=1)   # [B, 12, D]
            x_start = x_rep[:, 0:1, :]              # [B, 1, D]
            x_end = x_rep[:, -1:, :]                # [B, 1, D]
            return torch.cat([x_start, x_rep, x_end], dim=1)  # [B, 14, D]

        target_point_seq = expand_and_pad(batch['target_point_seq'])   # [B, 14, 3]
        ego_motion_full = expand_and_pad(batch['ego_motion_seq'])      # [B, 14, 3]
        acc_rew_seq = batch['acc_rew'].unsqueeze(-1)                   # [B, 4, 1]
        acc_rew_full = expand_and_pad(acc_rew_seq)                     # [B, 14, 1]

        # ---------------------------------------------------
        # Encode control tokens (discrete), project continuous
        # ---------------------------------------------------
        continuous_part = torch.cat([
            target_point_seq,   # [B, 14, 3]
            ego_motion_full,    # [B, 14, 3]
            acc_rew_full        # [B, 14, 1]
        ], dim=-1)              # → [B, 14, 7]

        control_embed = self.embedding(control_tokens.squeeze(-1))  # [B, 14, control_emb_dim]
        cont_proj = self.input_proj(continuous_part)                    # [B, 14, d_model - control_emb_dim]

        tgt_embedding = torch.cat([cont_proj, control_embed], dim=-1)   # [B, 14, d_model]

        # ---------------------------------------------------
        # Positional encoding & decoder forward
        # ---------------------------------------------------
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)
        # tgt_embedding = tgt_embedding.transpose(0, 1)  # [14, B, d_model]
        # memory = encoder_out.transpose(0, 1)           # [L_enc, B, d_model]

        # tgt_mask = self.generate_causal_mask(14).to(tgt_embedding.device)  # [14, 14]
        tgt_mask, tgt_padding_mask = self.create_mask(gt_control)

        decoder_out = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_controls = self.output(decoder_out)  # [14, B, vocab_size]

        return pred_controls     # [B, 14, vocab_size]


    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), self.cfg.tf_de_tgt_dim - length - 1).fill_(self.pad_idx).long().to('cuda')
        tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + self.pos_embed

        pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_controls_f = self.output(pred_controls)[:, length - 1, :]

        pred_controls = torch.softmax(pred_controls_f, dim=-1)
        pred_controls = pred_controls.argmax(dim=-1).view(-1, 1)
        return pred_controls

    def predict_logits(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), self.cfg.tf_de_tgt_dim - length - 1).fill_(self.pad_idx).long().to('cuda')
        tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + self.pos_embed

        pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_controls_f = self.output(pred_controls)[:, length - 1, :]

        pred_controls = torch.softmax(pred_controls_f, dim=-1)
        pred_controls = pred_controls.argmax(dim=-1).view(-1, 1)

        return pred_controls, pred_controls_f

class ControlPredictRL(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.control_policy_rl = CNNMLPPolicy(self.cfg, input_channels=self.cfg.tf_de_dim, num_actions=3, bins_per_action=self.cfg.token_nums)

    def forward(self, refined_feature):
        logits = self.control_policy_rl(refined_feature)  # [1, 3, 200]
        dists = [torch.distributions.Categorical(logits=logits[:, i]) for i in range(3)]
        actions = torch.stack([d.sample() for d in dists], dim=1)  # [1, 3]
        pred_multi_controls = torch.cat([pred_multi_controls, actions], dim=1)
        return pred_controls

    def predict(self, refined_feature):
        logits = self.control_policy_rl(refined_feature)  # [1, 3, 200]
        dists = [torch.distributions.Categorical(logits=logits[:, i]) for i in range(3)]
        actions = torch.stack([d.sample() for d in dists], dim=1)  # [1, 3]
        pred_multi_controls = torch.cat([pred_multi_controls, actions], dim=1)
        return pred_controls


class CNNMLPPolicy(nn.Module):
    def __init__(self, cfg, input_channels=264, num_actions=3, bins_per_action=200):
        super().__init__()

        # reshape (B, 256, 264) -> (B, 264, 16, 16)
        self.cfg = cfg
        self.input_tokens = self.cfg.tf_en_bev_length
        self.token_h = self.token_w = int(self.input_tokens**0.5)
        self.feature_dim = input_channels

        # CNN over 16x16 tokens
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),  # [B, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),              # [B, 64, 16, 16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                               # [B, 64, 1, 1]
        )

        # MLP to output multi-discrete logits
        self.mlp = nn.Sequential(
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions * bins_per_action)
        )

        self.num_actions = num_actions
        self.bins_per_action = bins_per_action

    def forward(self, x):  # x: [B, 256, 264]
        B = x.shape[0]
        x = x.view(B, self.token_h, self.token_w, self.feature_dim)   # [B, 16, 16, 264]
        x = x.permute(0, 3, 1, 2)                                     # [B, 264, 16, 16]

        x = self.cnn(x)                                               # [B, 64, 1, 1]
        logits = self.mlp(x)                                          # [B, num_actions * bins]
        return logits.view(B, self.num_actions, self.bins_per_action)  # [B, 3, 200]