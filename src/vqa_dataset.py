import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset
from student_code.vqa import VQA
import pickle
import torch
import numpy as np
from student_code.preprocess import utils
from PIL import Image
from student_code.preprocess.dataset import Dictionary
from torchvision import transforms

dataroot = os.path.abspath('data')

def _load_dataset(vqa, QuesIds, name):
    """Load entries
    """
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    QuesIds.sort()
    utils.assert_eq(len(QuesIds), len(answers))
    entries = []
    for quesId, ans in zip(QuesIds, answers):
        imgId = vqa.getImgIds(quesIds=quesId)[0]
        utils.assert_eq(quesId, ans['question_id'])
        utils.assert_eq(imgId, ans['image_id'])
        ans.pop('image_id')
        ans.pop('question_id')
        ques = vqa.qqa[quesId]['question']
        entry = {
            'question_id': quesId,
            'image_id': imgId,
            'question': ques,
            'answer': ans}
        entries.append(entry)
    return entries

def _load_img(image_path_pattern, image_id):
    path = image_path_pattern.format(image_id)
    img_pil = Image.open(path).convert("RGB")
    return img_pil

def scale_keep_ar_min_fixed(img, fixed_min):
    # Scale proportionally to let the shorter side to be fixed_min
    ow, oh = img.size
    if ow < oh:
        nw = fixed_min
        nh = nw * oh // ow
    else:
        nh = fixed_min
        nw = nh * ow // oh
    return img.resize((nw, nh), Image.BICUBIC)

def _get_transform(dim=256):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: scale_keep_ar_min_fixed(img, dim)))
    transform_list.append(transforms.CenterCrop((dim, dim)))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 debug=False):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self.image_path_pattern = os.path.join(image_dir, image_filename_pattern.replace('{}', '{:0>12}'))
        # print('\n question_json_file_path:', question_json_file_path)
        # print('\n annotation_json_file_path:', annotation_json_file_path)
        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.QuesIds = self.vqa.getQuesIds()
        if 'test_questions.json' in question_json_file_path:
            debug = True

        if debug:
            ans2label_path = os.path.join(dataroot, 'cache', 'only_debug_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 'only_debug_label2ans.pkl')
            self.entries = _load_dataset(self.vqa, self.QuesIds, 'debug')
        else:
            ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
            name = 'train' if 'train' in image_filename_pattern else 'val'
            self.entries = _load_dataset(self.vqa, self.QuesIds, name)

        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))
        self.tokenize()
        self.tensorize()
        self.preprocess = _get_transform()


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_pil = _load_img(self.image_path_pattern, entry['image_id'])
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        each_data = {'img': self.preprocess(img_pil), 'question': question, 'target': target}
        return each_data
