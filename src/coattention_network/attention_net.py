import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class coattention(nn.Module):
    """coattention

    get high-level h from given V (d*N) and Q (d*T).
    """
    def __init__(self, dim_d):
        super(coattention, self).__init__()
        dim_k = dim_d
        self.W_b = Parameter(torch.Tensor(dim_d, dim_d))
        self.W_v = Parameter(torch.Tensor(dim_k, dim_d))
        self.W_q = Parameter(torch.Tensor(dim_k, dim_d))
        self.w_hv = Parameter(torch.Tensor(1, dim_k))
        self.w_hq = Parameter(torch.Tensor(1, dim_k))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, Q, V):
        """
        :param Q: [batch, dim_d, dim_T]
        :param V: [batch, dim_d, dim_N]
        :return: q_hat [dim_d], v_hat [dim_d]
        """
        # print('\n Q:', Q.size())
        # print('\n V:', V.size())
        QT = torch.transpose(Q, 1, 2)
        C = QT.matmul(self.W_b.matmul(V))  # [dim_d, dim_d]
        C = self.tanh(C)
        # print('\n size C:', C.size())
        Hv = self.tanh(self.W_v.matmul(V) + self.W_q.matmul(Q).matmul(C))
        av = self.softmax(self.w_hv.matmul(Hv))
        v_hat = torch.bmm(av, V.transpose(1, 2)).squeeze()
        # print('\n v_hat:', v_hat.size())
        # print('\n size 1:', self.W_q.matmul(Q).size())
        # print('\n size 2 part:', self.W_v.matmul(V).transpose(0, 1).size())
        Hq = self.tanh(self.W_q.matmul(Q) + self.W_v.matmul(V).matmul(torch.transpose(C, 1, 2)))
        aq = self.softmax(self.w_hq.matmul(Hq))
        q_hat = torch.bmm(aq, Q.transpose(1, 2)).squeeze()
        return q_hat, v_hat