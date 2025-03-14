import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import torchvision

class FeatureExtractor(nn.Module):  # Inherit from nn.Module
    def __init__(self, model_name='facebook/dinov2-base', output_dim=512, image_size=224):
        super(FeatureExtractor, self).__init__()  # Initialize nn.Module
        self.model_name = model_name
        self.output_dim = output_dim
        self.image_size = image_size  # Initial assumption for positional embedding

        # Load model and initialize parameters
        if 'dinov2' in model_name.lower():
            self.model_type = 'dinov2'
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.feature_dim = self.model.config.hidden_size
            self.patch_size = self.model.config.patch_size
        elif model_name.lower() == 'resnet18':
            self.model_type = 'resnet18'
            resnet = torchvision.models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            self.feature_dim = 512
            self.patch_size = 32  # ResNet's final downsampling factor
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Calculate initial patches_per_side (assuming square input)
        self.patches_per_side = self.image_size // self.patch_size
        
        # Projection layer
        self.input_proj = nn.Linear(self.feature_dim, output_dim)
        
        # Positional embedding (will be adapted during forward if needed)
        self.pos_embed = nn.Parameter(torch.randn(1, output_dim, 1, 1))  # Initialized small

    def _process_resnet(self, images):
        """Process images for ResNet18 with dynamic size handling"""
        # Convert from [0,255] to [0,1] and normalize
        images = images.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    def forward(self, images):
        # Input size debugging
        print(f"Input size: {tuple(images.shape)}")

        batch_size, num_images, channels, height, width = images.shape

        # Reshape and concatenate along the width
        images_total = images.permute(0, 2, 3, 1, 4).reshape(batch_size, channels, height, num_images * width)

        
        B, C, H, W = images_total.shape
        
        # Feature extraction
        if self.model_type == 'dinov2':
            inputs = self.processor(
                images=images_total,
                return_tensors="pt",
                do_rescale=True,
                do_normalize=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            features = outputs.last_hidden_state  # [B, num_patches, feat_dim]
            h = w = int(features.shape[1] ** 0.5)
            features = features.view(B, h, w, -1).permute(0, 3, 1, 2)
        else:
            processed_images = self._process_resnet(images_total)
            features = self.model(processed_images)  # [B, 512, h, w]
        
        # Feature map size after backbone
        print(f"Backbone output size: {features.shape}")
        
        # Adaptive projection
        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        features = self.input_proj(features)
        features = features.permute(0, 3, 1, 2)  # [B, output_dim, H, W]
        
        # Adaptive positional embedding
        _, _, fh, fw = features.shape
        if self.pos_embed.shape[2:] != (fh, fw):
            # Dynamically create new positional embedding
            self.pos_embed = nn.Parameter(torch.randn(1, self.output_dim, fh, fw))
            print(f"Updated positional embedding size: {self.pos_embed.shape}")
            
        features += self.pos_embed.to(features.device)
        
        # Final feature map size
        print(f"Final feature map size: {features.shape}")
        return features