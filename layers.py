import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn


class AttLayer2(nn.Module):
    """Soft alignment attention implement.

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=768, seed=0): # PARCHE
        """Initialization steps for AttLayer2.

        Args:
            dim (int): attention hidden dim
        """
        super(AttLayer2, self).__init__()
        self.dim = dim
        self.seed = seed
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.Tensor(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.q = nn.Parameter(torch.Tensor(dim, 1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs, mask=None):
        """Core implementation of soft attention

        Args:
            inputs (object): input tensor.

        Returns:
            object: weighted sum of input tensors.
        """
        attention = torch.tanh(torch.matmul(inputs, self.W) + self.b)
        attention = torch.matmul(attention, self.q).squeeze(2)

        if mask is None:
            attention = torch.exp(attention)
        else:
            attention = torch.exp(attention) * mask.float()

        attention_weight = attention / (torch.sum(attention, dim=-1, keepdim=True) + 1e-8)
        attention_weight = attention_weight.unsqueeze(2)
        weighted_input = inputs * attention_weight
        return torch.sum(weighted_input, dim=1)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor

        Args:
            input_shape (tuple): shape of input tensor.

        Returns:
            tuple: shape of output tensor.
        """
        return input_shape[0], input_shape[-1]

class SelfAttention(nn.Module):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        """Initialization steps for AttLayer2.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimention of each head.
            mask_right (boolean): whether to mask right words.
        """
        super(SelfAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        self.WQ = nn.Linear(self.output_dim, self.head_dim)
        self.WK = nn.Linear(self.output_dim, self.head_dim)
        self.WV = nn.Linear(self.output_dim, self.head_dim)

    def forward(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. qeury, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        Q_seq = self.WQ(Q_seq)
        Q_seq = Q_seq.view(-1, Q_seq.size(1), self.multiheads, self.head_dim)
        Q_seq = Q_seq.permute(0, 2, 1, 3)

        K_seq = self.WK(K_seq)
        K_seq = K_seq.view(-1, K_seq.size(1), self.multiheads, self.head_dim)
        K_seq = K_seq.permute(0, 2, 1, 3)

        V_seq = self.WV(V_seq)
        V_seq = V_seq.view(-1, V_seq.size(1), self.multiheads, self.head_dim)
        V_seq = V_seq.permute(0, 2, 1, 3)
        A = torch.matmul(Q_seq, K_seq.transpose(2, 3)) / np.sqrt(self.head_dim)

        A = A.permute(0, 3, 2, 1)

        A = self.Mask(A, V_len, "add")
        A = A.permute(0, 3, 2, 1)

        if self.mask_right:
            ones = torch.ones_like(A[:1, :1])
            lower_triangular = torch.tril(ones)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = F.softmax(A, dim=-1)

        O_seq = torch.matmul(A, V_seq)
        O_seq = O_seq.permute(0, 2, 1, 3)

        O_seq = O_seq.view(-1, O_seq.size(1), self.output_dim)
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq
    
    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): Sequence length of inputs.
            mode (str): Mode of mask.

        Returns:
            object: Tensors after masking.
        """
        if seq_len is None:
            return inputs
        else:
            mask = F.one_hot(seq_len[:, 0], num_classes=inputs.size(1)).float()
            mask = 1 - torch.cumsum(mask, dim=1)

            for _ in range(len(inputs.shape) - 2):
                mask = mask.unsqueeze(2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def get_config(self):
        """Add multiheads, head_dim and mask_right into layer config.

        Returns:
            dict: Config of SelfAttention layer.
        """
        config = {
            "multiheads": self.multiheads,
            "head_dim": self.head_dim,
            "mask_right": self.mask_right,
        }
        return config
