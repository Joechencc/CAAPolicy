import torch
import numpy as np

from torch import nn
from tool.config import Configuration
from dataset.carla_dataset import detokenize_control_group


class ControlLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlLoss, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, pred, data):
        pred_control = pred.reshape(-1, pred.shape[-1])
        gt_control = data['gt_control'][:, 1:].reshape(-1).cuda()
        control_loss = self.ce_loss(pred_control, gt_control)
        return control_loss


class ControlValLoss(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlValLoss, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1

        self.valid_token = self.cfg.token_nums - 4
        self.half_token = float(self.valid_token / 2)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.l1_loss = nn.SmoothL1Loss()

    def detokenize_acc(self, acc_token):
        if acc_token > self.half_token:
            acc = (acc_token / self.half_token - 1)
        else:
            acc = -(acc_token / self.half_token - 1)
        return acc

    def detokenize_steer(self, steer_token):
        steer = (steer_token / self.half_token) - 1
        return steer

    def forward(self, pred, data):
        # Get predicted tokens and convert them to control values
        pred_control = pred[:, :-2, :]  # Exclude BOS, EOS
        pred_token = torch.softmax(pred_control, dim=-1).argmax(dim=-1)
        
        # Create empty lists to store detokenized values
        pred_acc_list = []
        pred_steer_list = []
        pred_reverse_list = []
        
        # Detokenize each token in the batch
        for batch_idx in range(pred_token.size(0)):
            for seq_idx in range(pred_token.size(1)):
                token = pred_token[batch_idx, seq_idx].item()
                
                # Skip padding tokens
                if token == self.pad_idx:
                    continue
                    
                # Use your detokenize_control_group function
                controls = detokenize_control_group(token, self.cfg.token_nums)
                
                # Extract individual control values
                # controls = [throttle, brake, steer, reverse]
                throttle_brake = controls[0] - controls[1]  # throttle - brake
                steer = controls[2]
                reverse = 1 if controls[3] else 0
                
                pred_acc_list.append(throttle_brake)
                pred_steer_list.append(steer)
                pred_reverse_list.append(reverse)
        
        # Convert lists to tensors
        pred_acc = torch.tensor(pred_acc_list, dtype=torch.float32).cuda()
        pred_steer = torch.tensor(pred_steer_list, dtype=torch.float32).cuda()
        pred_reverse = torch.tensor(pred_reverse_list, dtype=torch.int64).cuda()
        
        # Make sure ground truth is the right length
        gt_acc = data['gt_acc'].reshape(-1).cuda()[:len(pred_acc)]
        gt_steer = data['gt_steer'].reshape(-1).cuda()[:len(pred_steer)]
        gt_reverse = data['gt_reverse'].reshape(-1).cuda()[:len(pred_reverse)]
        
        # Calculate losses
        acc_val_loss = self.l1_loss(pred_acc, gt_acc)
        steer_val_loss = self.l1_loss(pred_steer, gt_steer)
        acc_steer_val_loss = (acc_val_loss + steer_val_loss)
        
        # For reverse, we need one-hot encoding for cross entropy
        pred_reverse_one_hot = torch.zeros(pred_reverse.size(0), 2).cuda()
        pred_reverse_one_hot[torch.arange(pred_reverse.size(0)), pred_reverse] = 1.0
        
        reverse_val_loss = self.ce_loss(pred_reverse_one_hot, gt_reverse)
        
        return acc_steer_val_loss, reverse_val_loss

    # def forward(self, pred, data):
    #     pred_control = pred[:, :-2, :]
    #     pred_acc_token = pred_control[:, 0::3, :]
    #     pred_steer_token = pred_control[:, 1::3, :]
    #     pred_reverse_token = pred_control[:, 2::3, :]

    #     pred_acc_token = torch.softmax(pred_acc_token, dim=-1)
    #     pred_acc_token = pred_acc_token.argmax(dim=-1)
    #     pred_acc_token = pred_acc_token.reshape(-1).tolist()
    #     pred_acc = [self.detokenize_acc(x) for x in pred_acc_token]
    #     pred_acc = torch.from_numpy(np.array(pred_acc, dtype=np.float32)).cuda()
    #     gt_acc = data['gt_acc'].reshape(-1).cuda()
    #     acc_val_loss = self.l1_loss(pred_acc, gt_acc)

    #     pred_steer_token = torch.softmax(pred_steer_token, dim=-1)
    #     pred_steer_token = pred_steer_token.argmax(dim=-1)
    #     pred_steer_token = pred_steer_token.reshape(-1)
    #     pred_steer = self.detokenize_steer(pred_steer_token)
    #     gt_steer = data['gt_steer'].reshape(-1).cuda()
    #     steer_val_loss = self.l1_loss(pred_steer, gt_steer)

    #     acc_steer_val_loss = (acc_val_loss + steer_val_loss)

    #     pred_reverse_token = torch.softmax(pred_reverse_token, dim=-1)
    #     p_no_reverse = torch.sum(pred_reverse_token[:, :, :101], dim=-1).reshape(-1)
    #     p_reverse = torch.sum(pred_reverse_token[:, :, 101:], dim=-1).reshape(-1)
    #     pred_reverse = torch.stack((p_no_reverse, p_reverse), dim=0).T
    #     gt_reverse = data['gt_reverse'].reshape(-1).cuda()
    #     reverse_val_loss = self.ce_loss(pred_reverse, gt_reverse)

    #    return acc_steer_val_loss, reverse_val_loss
