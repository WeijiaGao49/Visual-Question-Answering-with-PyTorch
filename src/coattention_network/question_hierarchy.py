import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class PhraseFeature(nn.Module):
    """PhraseFeature
    get PhraseFeature Qp (d*T) from given  Qw (d*T).
    """
    def __init__(self, dim_d):
        super(PhraseFeature, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.tanh = nn.Tanh()
        self.Ws_c = nn.Linear(dim_d, dim_d, bias=False)
        # self.Ws_c = Parameter(torch.Tensor(dim_d, dim_d))

    def forward(self, Qw):
        """
        :param Qw: [batch, d, T]
        :return: Qp: [batch, d, T]
        """
        batch, d, T = Qw.size()
        Qw = Qw.contiguous().view(batch * d, 1, T)

        conv_out = self.conv1(Qw).view(batch, d, T).transpose(1, 2)
        Qw1_p = self.tanh(self.Ws_c(conv_out)).view(batch, T, d, 1)

        conv_out = self.conv2(Qw).view(batch, d, T).transpose(1, 2)
        Qw2_p = self.tanh(self.Ws_c(conv_out)).view(batch, T, d, 1)

        conv_out = self.conv3(Qw).view(batch, d, T).transpose(1, 2)
        Qw3_p = self.tanh(self.Ws_c(conv_out)).view(batch, T, d, 1)

        Qp, _ = torch.max(torch.cat((Qw1_p, Qw2_p, Qw3_p), dim=3), dim=3)
        # print('\n Qp:', Qp.size())
        return Qp.transpose(1, 2)

class QuestionFeature(nn.Module):
    """QuestionFeature
    get QuestionFeature Qs (d*T) from given  Qp (d*T).
    """
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout):
        super(QuestionFeature, self).__init__()
        self.rnn = nn.LSTM(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.ndirections = 1 + int(bidirect)

    def forward(self, Qp):
        Qs, _ = self.rnn(Qp.transpose(1, 2))
        # print('\n Qs', Qs.size())
        return Qs.transpose(1, 2)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        return (Variable(weight.new(*hid_shape).zero_()),
                Variable(weight.new(*hid_shape).zero_()))