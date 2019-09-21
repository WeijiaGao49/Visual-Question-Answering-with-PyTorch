import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from student_code.preprocess.classifier import SimpleClassifier
from student_code.googlenet_fea import googlenet
from student_code.language_model import WordEmbedding
from student_code.coattention_network.question_hierarchy import QuestionFeature, PhraseFeature
from student_code.coattention_network.attention_net import coattention
import torch
import math

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, num_ans_candidates, ntoken):
        super(CoattentionNet, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        dim_img_fea = 1024
        dim_N = 1
        dim_ques_fea = 300
        dim_hid = 300
        self.dim_N = dim_N
        self.dim_hid = dim_hid
        self.img_linear = nn.Linear(dim_img_fea, dim_hid * dim_N)
        self.w_fea = WordEmbedding(ntoken, 300, 0.0)
        self.p_fea = PhraseFeature(300)
        self.s_fea = QuestionFeature(300, dim_ques_fea, 1, False, 0.0)
        self.tanh = nn.Tanh()
        self.linear_w = nn.Linear(dim_hid, dim_hid, bias=False)
        self.linear_p = nn.Linear(2 * dim_hid, dim_hid, bias=False)
        self.linear_s = nn.Linear(2 * dim_hid, dim_hid, bias=False)
        # self.linear_Pre = nn.Linear(dim_hid, num_ans_candidates, bias=False)
        # self.softmax = nn.Softmax()
        self.att_w = coattention(dim_hid)
        self.att_p = coattention(dim_hid)
        self.att_s = coattention(dim_hid)
        self.classifier = SimpleClassifier(
            2 * dim_hid + dim_img_fea, 2 * dim_hid, num_ans_candidates, 0.5)
        self._initialize_weights()

    def forward(self, image, question_encoding):
        batch_size = image.size(0)
        img_fea = self.googlenet.feature(image)
        V = self.img_linear(img_fea).view(batch_size, self.dim_hid, self.dim_N)
        Qw = self.w_fea(question_encoding).transpose(1, 2)
        Qp = self.p_fea(Qw)
        Qs = self.s_fea(Qp)
        q_hat_w, v_hat_w = self.att_w(Qw, V)
        q_hat_p, v_hat_p = self.att_p(Qp, V)
        q_hat_s, v_hat_s = self.att_p(Qs, V)
        hw = self.tanh(self.linear_w(q_hat_w + v_hat_w))
        # print('\n size linear:', torch.cat((q_hat_p + v_hat_p, hw), dim=1).size())
        hp = self.tanh(self.linear_p(torch.cat((q_hat_p + v_hat_p, hw), dim=1)))
        hs = self.tanh(self.linear_s(torch.cat((q_hat_s + v_hat_s, hp), dim=1)))
        p = self.classifier(torch.cat((hs, Qs[:,:,-1].squeeze(), img_fea), dim=1))
        return p

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

