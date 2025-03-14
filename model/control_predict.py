import torch

from torch import nn
from timm.models.layers import trunc_normal_
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1

        # Embedding for both image and control tokens
        self.control_embedding = nn.Embedding(self.cfg.token_nums, self.cfg.tf_de_dim)
        self.img_embedding = nn.Embedding(self.cfg.img_feature_dim, self.cfg.tf_de_dim) # for discrete token not image token
        self.img_projection = nn.Linear(self.cfg.img_feature_dim, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)

        # Positional embeddings for the combined sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.cfg.tf_de_img_length + self.cfg.tf_de_control_length, self.cfg.tf_de_dim) * .02
        )

        # Transformer decoder
        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)

        # Separate output layers for image and control tokens
        self.output_image = nn.Linear(self.cfg.tf_de_dim, self.cfg.img_feature_dim)  # For image tokens (output size img_feature_dim)
        self.output_control = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums)  # For control tokens

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    # def create_mask(self, tgt):
    #     # Create causal mask for the combined sequence
    #     seq_len = tgt.shape[1]
    #     tgt_mask = (torch.triu(torch.ones((seq_len, seq_len), device=self.cfg.device)) == 1).transpose(0, 1)
    #     tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

    #     # Create padding mask for the combined sequence
    #     tgt_padding_mask = (tgt == self.pad_idx)
    #     return tgt_mask, tgt_padding_mask
    def create_mask(self, tgt):
        """
        Create masks for the target sequence.

        Args:
            tgt: Concatenated sequence of img_feature and control_embed.
                Shape: (batch_size, sequence_length, feature_dim)

        Returns:
            tgt_mask: Causal mask for autoregressive decoding.
                    Shape: (sequence_length, sequence_length)
            tgt_padding_mask: Padding mask indicating padding tokens.
                            Shape: (batch_size, sequence_length)
        """
        # Create causal mask for the combined sequence
        seq_len = tgt.shape[1]
        tgt_mask = (torch.triu(torch.ones((seq_len, seq_len), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        # Create padding mask for the combined sequence
        # Assumption: Padding is indicated by zeros in the first feature dimension
        tgt_padding_mask = (tgt[:, :, 0] == 0)  # (batch_size, sequence_length)

        return tgt_mask, tgt_padding_mask



    def decoder(self, tgt_embedding, encoder_out, tgt_mask, tgt_padding_mask):
        # Transpose inputs for transformer decoder
        encoder_out = encoder_out.transpose(0, 1)  # (B, S_enc, D) -> (S_enc, B, D)
        tgt_embedding = tgt_embedding.transpose(0, 1)  # (B, S_tgt, D) -> (S_tgt, B, D)

        # Pass through transformer decoder
        pred_tokens = self.tf_decoder(
            tgt=tgt_embedding,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # Transpose back to batch-first format
        pred_tokens = pred_tokens.transpose(0, 1)  # (S_tgt, B, D) -> (B, S_tgt, D)
        return pred_tokens

    def forward(self, encoder_out, control, img_feature):
        # Print input shapes
        print(f"Input shapes:")
        print(f"encoder_out shape: {encoder_out.shape}")
        print(f"control shape: {control.shape}")
        print(f"img_feature shape: {img_feature.shape}")

        # Flatten image features and project
        B, C, H, W = img_feature.shape
        img_feature = img_feature.permute(0, 2, 3, 1).reshape(B, H * W, C)
        img_feature = self.img_projection(img_feature)  # (B, H*W, D)

        # Add BOS token (ID=201) to the start of the sequence
        BOS_TOKEN = 201
        bos = torch.full((B, 1), BOS_TOKEN, device=control.device)
        bos_embed = self.control_embedding(bos)  # (B, 1, D)

        # Remove old BOS and last token from control
        control = control[:, 1:-1]  # (B, 13)
        control_embed = self.control_embedding(control)  # (B, 13, D)

        # Concatenate: [BOS][Image Features][Control Tokens]
        combined_sequence = torch.cat([bos_embed, img_feature, control_embed], dim=1)  # (B, 1 + 384 + 13, D)

        # Create masks for the full sequence (including BOS)
        tgt_mask, tgt_padding_mask = self.create_mask(combined_sequence)

        # Debugging: Print mask shapes
        print(f"tgt_mask shape: {tgt_mask.shape}")
        print(f"tgt_padding_mask shape: {tgt_padding_mask.shape}")

        # Apply positional embeddings to the entire sequence
        seq_len = combined_sequence.size(1)
        pos_embed = self.pos_embed[:, :seq_len, :]  # (1, seq_len, D)
        tgt_embedding = self.pos_drop(combined_sequence + pos_embed)

        # Decode the sequence (BOS + img + control)
        pred_tokens = self.decoder(tgt_embedding, encoder_out, tgt_mask, tgt_padding_mask)

        # Split predictions
        pred_image_tokens = pred_tokens[:, :self.cfg.tf_de_img_length, :] 
        pred_control_tokens = pred_tokens[:, self.cfg.tf_de_img_length:, :]

        # Project to output dimensions
        pred_image_tokens = self.output_image(pred_image_tokens)
        pred_control_tokens = self.output_control(pred_control_tokens)

        return pred_image_tokens, pred_control_tokens
    
    def predict(self, encoder_out, BOS_token):
        # Initialize the target sequence with BOS token and padding
        batch_size = encoder_out.size(0)
        total_length = self.cfg.tf_de_img_length + self.cfg.tf_de_control_length  # Total sequence length (image + control tokens)
        
        # Initialize target sequence: BOS token at the start, followed by padding
        tgt = torch.full((batch_size, total_length), self.pad_idx, dtype=torch.long, device=self.cfg.device)
        tgt[:, 0] = BOS_token  # Set the first token as BOS

        # Create masks for the target sequence
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        # Embed the initial target sequence
        control_embed = self.control_embedding(tgt[:, self.cfg.tf_de_img_length:])  # Embed control portion
        control_embed = self.pos_drop(control_embed + self.pos_embed[:, self.cfg.tf_de_img_length:, :])
        
        img_embed = self.img_embedding(tgt[:, :self.cfg.tf_de_img_length])  # Embed image portion (initially padded)
        img_embed = self.pos_drop(img_embed + self.pos_embed[:, :self.cfg.tf_de_img_length, :])
        
        # Concatenate embeddings
        tgt_embedding = torch.cat([img_embed, control_embed], dim=1)

        # Decode the full sequence in one pass
        pred_tokens = self.decoder(tgt_embedding, encoder_out, tgt_mask, tgt_padding_mask)

        # Split the output sequence into image and control tokens
        pred_image_tokens = pred_tokens[:, :self.cfg.tf_de_img_length, :]
        pred_control_tokens = pred_tokens[:, self.cfg.tf_de_img_length:, :]

        # Predict image feature tokens
        pred_image_tokens = self.output_image(pred_image_tokens)

        # Predict control tokens
        pred_control_tokens = self.output_control(pred_control_tokens)

        # Apply softmax and argmax to get the most likely control tokens
        pred_control = torch.softmax(pred_control_tokens, dim=-1).argmax(dim=-1) 
        pred_control = pred_control[:3] # take first 3 for the current time step
        
        return pred_image_tokens, pred_control
    
    # ---------------Sequential version--------------:
    # def predict(self, encoder_out, BOS_token): 
    #     # initialize tgt with BOS_token = 201
    #     tgt = BOS_token
    #     pred_control = torch.ones(3)

    #     for step in range(self.cfg.tf_de_img_length + 3):  # 256 image tokens + 3 control tokens
    #         padding = torch.ones(tgt.size(0), self.cfg.tf_de_img_length + self.cfg.tf_de_control_length - step - 1).fill_(self.pad_idx).long().to('cuda')
    #         if step == 0:
    #             tgt = torch.cat([tgt, padding], dim=1)
    #             # Create masks for tgt
    #             tgt_mask, tgt_padding_mask = self.create_mask(tgt)

    #             # Embed the control tokens
    #             control_embed = self.control_embedding(tgt[self.cfg.tf_de_img_length:])
    #             control_embed = control_embed + self.pos_embed[:, self.cfg.tf_de_img_length:, :]

    #             # Embed the image feature tokens
    #             img_embed = self.img_embedding(tgt[:self.cfg.tf_de_img_length])
    #             img_embed = img_embed + self.pos_embed[:, :self.cfg.tf_de_img_length, :]
    #             # Concatenate the embeddings
    #             tgt_embedding = torch.cat([img_embed, control_embed], dim=1)

    #         # Remember to update tgt_mask and tgt_padding mask

    #         # Decode the combined sequence
    #         pred_tokens = self.decoder(tgt_embedding, encoder_out, tgt_mask, tgt_padding_mask)

    #         # Split the output sequence into image tokens and control tokens
    #         pred_image_tokens = pred_tokens[:, :self.cfg.tf_de_img_length, :]  # (B, S_img, D)
    #         pred_control_tokens = pred_tokens[:, self.cfg.tf_de_img_length:, :]  # (B, S_control, D)

    #         # Predict image feature tokens (size img_feature_dim)
    #         pred_image_tokens = self.output_image(pred_image_tokens)  # (B, S_img, img_feature_dim)

    #         # Predict control tokens
    #         pred_control_tokens = self.output_control(pred_control_tokens)  # (B, S_control, token_nums)

    #         # Update img_feature and control for the next step
    #         if step >= self.cfg.tf_de_img_length:
    #             # update tgt_embedding as the input of the next step autoregressive prediction
    #             img_embed = self.img_embedding(pred_control_tokens)
    #             control_embed = self.control_embedding(pred_image_tokens)
    #             tgt_embedding = torch.cat([img_embed, control_embed], dim=1)
                
    #             # TODO: pad the embedding
    #             # Calculate the required padding size
    #             padding_size = (self.cfg.tf_de_img_length + self.cfg.tf_de_control_length) - tgt_embedding.size(1)
    #             if padding_size > 0:
    #                 # Create padding tensor with zeros (or another appropriate value)
    #                 padding_embed = torch.zeros((tgt_embedding.size(0), padding_size, tgt_embedding.size(2)), device=tgt_embedding.device)
    #                 tgt_embedding = torch.cat([tgt_embedding, padding_embed], dim=1)

    #             # save control tokens
    #             pred_control[:, step - self.cfg.tf_de_img_length] = torch.softmax(pred_control_tokens[:, step - self.cfg.tf_de_img_length, :], dim=-1).argmax(dim=-1)
    #         else:
    #             tgt_embedding = self.img_embedding(pred_image_tokens)
                
    #             # TODO: pad the embedding
    #             # Calculate the required padding size
    #             padding_size = (self.cfg.tf_de_img_length + self.cfg.tf_de_control_length) - tgt_embedding.size(1)
    #             if padding_size > 0:
    #                 # Create padding tensor with zeros (or another appropriate value)
    #                 padding_embed = torch.zeros((tgt_embedding.size(0), padding_size, tgt_embedding.size(2)), device=tgt_embedding.device)
    #                 tgt_embedding = torch.cat([tgt_embedding, padding_embed], dim=1)

    #     return pred_control