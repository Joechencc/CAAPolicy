import torch
from torch import nn
from model.DINOv2_extractor import DINOv2FeatureExtractor
from timm.models.layers import trunc_normal_
from tool.config import Configuration

class Vision_ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1

        # DINOv2 for image tokenization
        self.dinov2 = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-base",
            output_dim=self.cfg.tf_de_dim  # Ensure output matches decoder dim
        )
        
        # Project DINOv2 tokens to embedding space
        self.image_token_proj = nn.Linear(
            self.dinov2.feature_dim,  # e.g., 768 for dinov2-base
            self.cfg.tf_de_dim
        )
        
        # Control token embedding (original)
        self.embedding = nn.Embedding(self.cfg.token_nums, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)
        
        # Positional embedding for combined sequence (image + control tokens)
        self.combined_seq_len = self.dinov2.image_token_seq_len + (self.cfg.tf_de_tgt_dim - 1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.combined_seq_len, self.cfg.tf_de_dim) * .02
        )

        # Prediction heads
        self.image_feature_head = nn.Linear(self.cfg.tf_de_dim, self.dinov2.feature_dim)
        self.control_head = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums)

        # Transformer decoder (unchanged)
        tf_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.tf_de_dim,
            nhead=self.cfg.tf_de_heads
        )
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
        
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'dinov2' in name:
                p.requires_grad_(False)
            elif 'pos_embed' in name:  # Skip pos_embed here
                continue
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)  # Explicit init for pos_embed

    def create_mask(self, seq_len):
        mask = (torch.triu(torch.ones((seq_len, seq_len), device=self.cfg.device)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, encoder_out, images, tgt):
        # Extract DINOv2 image tokens
        image_features = self.dinov2(images)  # (B, D, H, W)
        B, D, H, W = image_features.shape
        image_tokens = image_features.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
        image_tokens = self.image_token_proj(image_tokens)  # (B, num_image_tokens, tf_de_dim)

        # Prepare control tokens
        tgt = tgt[:, :-1]
        control_embed = self.embedding(tgt)  # (B, control_seq_len-1, tf_de_dim)
        
        # Concatenate image + control tokens
        combined_tgt = torch.cat([image_tokens, control_embed], dim=1)  # (B, total_seq_len, tf_de_dim)
        
        # Positional embedding
        combined_tgt = self.pos_drop(combined_tgt + self.pos_embed)

        # Create mask for combined sequence
        tgt_mask = self.create_mask(combined_tgt.shape[1])
        tgt_padding_mask = (torch.cat([
            torch.zeros_like(image_tokens[:, :, 0]),  # Image tokens never padded
            (tgt == self.pad_idx)
        ], dim=1))

        # Transformer decoder
        decoder_out = self.tf_decoder(
            tgt=combined_tgt.transpose(0, 1),  # (seq_len, B, dim)
            memory=encoder_out.transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        ).transpose(0, 1)  # (B, seq_len, dim)

        # Split outputs
        pred_image = self.image_feature_head(decoder_out[:, :image_tokens.shape[1]])
        pred_control = self.control_head(decoder_out[:, image_tokens.shape[1]:])

        return pred_image, pred_control
    
    def predict(self, encoder_out, images, init_tgt):
        # Extract DINOv2 image tokens (same as forward)
        image_features = self.dinov2(images)  # (B, D, H, W)
        B, D, H, W = image_features.shape
        image_tokens = image_features.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
        image_tokens = self.image_token_proj(image_tokens)  # (B, num_img_tokens, tf_de_dim)
        num_img_tokens = image_tokens.shape[1]

        # Pad control sequence to max expected length
        max_control_len = self.cfg.tf_de_tgt_dim - 1  # Assume fixed-length generation
        current_tgt = init_tgt  # (B, start_len), e.g., [START] token
        padding = torch.full((B, max_control_len - current_tgt.shape[1]), 
                            self.pad_idx, 
                            device=current_tgt.device)
        padded_tgt = torch.cat([current_tgt, padding], dim=1)  # (B, max_control_len)

        # Embed control tokens
        control_embed = self.embedding(padded_tgt)  # (B, max_control_len, tf_de_dim)

        # Concatenate image + control tokens
        combined_tgt = torch.cat([image_tokens, control_embed], dim=1)  # (B, total_seq_len, tf_de_dim)
        
        # Positional embedding (fixed for precomputed length)
        combined_tgt = combined_tgt + self.pos_embed  # Requires pos_embed to match total_seq_len

        # Create masks
        tgt_mask = self.create_mask(combined_tgt.shape[1])
        tgt_padding_mask = torch.cat([
            torch.zeros_like(image_tokens[:, :, 0], dtype=torch.bool),  # Image tokens not padded
            (padded_tgt == self.pad_idx)  # Mask padded control tokens
        ], dim=1)

        # Transformer decoder
        decoder_out = self.tf_decoder(
            tgt=combined_tgt.transpose(0, 1),  # (seq_len, B, dim)
            memory=encoder_out.transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        ).transpose(0, 1)  # (B, seq_len, dim)

        # Extract control predictions (ignore image tokens)
        pred_control = self.control_head(decoder_out[:, num_img_tokens:])  # (B, max_control_len, vocab)
        
        # Autoregressive decoding ???
        pred_tokens = []
        for i in range(current_tgt.shape[1], max_control_len):
            # Get logits for the i-th position (current token to predict)
            logits = pred_control[:, i, :]  # (B, vocab)
            # Greedy decoding (customize with sampling if needed)
            next_token = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)  # (B,)
            pred_tokens.append(next_token.unsqueeze(1))
        
        # Combine initial tokens and predictions
        full_pred = torch.cat([current_tgt] + pred_tokens, dim=1)  # (B, max_control_len)
        return full_pred