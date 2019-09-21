import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from student_code.preprocess.classifier import SimpleClassifier
from student_code.googlenet_fea import googlenet
from student_code.language_model import WordEmbedding, QuestionEmbedding
import torch


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, num_ans_candidates, ntoken):
        super(SimpleBaselineNet, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        dim_img_fea = 1024
        dim_ques_fea = 1024
        dim_hid = 1024
        self.w_emb = WordEmbedding(ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, dim_ques_fea, 1, False, 0.0)
        self.classifier = SimpleClassifier(
            dim_img_fea + dim_ques_fea, 2 * dim_hid, num_ans_candidates, 0.5)

    def forward(self, image, question_encoding):
        img_fea = self.googlenet.feature(image)
        w_emb = self.w_emb(question_encoding)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        pre_ans = self.classifier(torch.cat((img_fea, q_emb), dim=1))
        return pre_ans
