from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch.nn as nn
import torch

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, args):
        if not args.debug:
            train_dataset = VqaDataset(image_dir=train_image_dir,
                                       question_json_file_path=train_question_path,
                                       annotation_json_file_path=train_annotation_path,
                                       image_filename_pattern="COCO_train2014_{}.jpg",)
            val_dataset = VqaDataset(image_dir=test_image_dir,
                                     question_json_file_path=test_question_path,
                                     annotation_json_file_path=test_annotation_path,
                                     image_filename_pattern="COCO_val2014_{}.jpg",)
            model = SimpleBaselineNet(num_ans_candidates=2185, ntoken=train_dataset.dictionary.ntoken)
        else:
            train_dataset = VqaDataset(image_dir=train_image_dir,
                                       question_json_file_path=train_question_path,
                                       annotation_json_file_path=train_annotation_path,
                                       image_filename_pattern="COCO_train2014_{}.jpg",
                                       debug=True)
            val_dataset = train_dataset
            model = SimpleBaselineNet(num_ans_candidates=2, ntoken=train_dataset.dictionary.ntoken)
        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs,
                         num_data_loader_workers, args.use_cuda, 'simple_baseline')
        self.optim = torch.optim.Adamax(self._model.parameters())

    def _optimize(self, predicted_answers, true_answers):
        loss = self._instance_bce_with_logits(predicted_answers, true_answers)
        loss.backward()
        nn.utils.clip_grad_norm(self._model.parameters(), 0.25)
        self.optim.step()
        self.optim.zero_grad()
        return loss
