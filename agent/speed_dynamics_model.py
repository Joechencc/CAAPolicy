import torch
from collections import OrderedDict
from model.dynamics_model import DynamicsModel
import logging

class SpeedDynamicsModel:
    """
    A class to load the dynamics model and run inference.
    """
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        model = DynamicsModel(hidden_dim=128, output_dim=2)
        
        try:
            logging.info(f"Loading checkpoint from {self.model_path}")
            ckpt = torch.load(self.model_path, map_location=self.device)
        except FileNotFoundError:
            logging.error(f"Checkpoint file not found at {self.model_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise

        state_dict = ckpt.get('state_dict', ckpt)

        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k.replace('model.', '')] = v
            elif k.startswith('dynamics_model.'):
                clean_state_dict[k.replace('dynamics_model.', '')] = v
            else:
                clean_state_dict[k] = v
        
        try:
            model.load_state_dict(clean_state_dict)
            model.to(self.device)
            model.eval()
            logging.info('Successfully loaded Speed Dynamics Model.')
            return model
        except RuntimeError as e:
            logging.error(f"Error loading state dict into DynamicsModel: {e}")
            logging.error("The model architecture might not match the checkpoint. "
                          "Please ensure the model definition matches the saved weights.")
            raise

    def predict(self, data):
        if self.model:
            with torch.no_grad():
                return self.model(data)
        else:
            logging.warning("Prediction skipped because model is not loaded.")
            return None 