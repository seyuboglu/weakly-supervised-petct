"""
"""
import torch 
import torch.nn as nn 


class AttentionAggregator(nn.Module):
    
    def __init__(self, encoding_size=1024):
        """
        """
        super().__init__()
        self.att_projection = nn.Linear(in_features=encoding_size, 
                                        out_features=1,
                                        bias=True)
    
    def forward(self, encodings):
        """
        """
        encoding_proj = self.att_projection(encodings).squeeze(2)
        alpha = torch.nn.functional.softmax(encoding_proj, dim=-1)
        aggregated_encodings = torch.bmm(alpha.unsqueeze(1), encodings).squeeze(1)
        return aggregated_encodings
