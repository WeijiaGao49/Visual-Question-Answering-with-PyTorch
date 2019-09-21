from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10,
                 use_cuda=True, runner_name=''):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 1000  # Steps
        self._runner_name = runner_name

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = use_cuda

        if self._cuda:
            self._model = self._model.cuda()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def _instance_bce_with_logits(self, logits, labels):
        assert logits.dim() == 2
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss *= labels.size(1)
        return loss

    def _compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size())
        if self._cuda:
            one_hots = one_hots.cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        labels = (labels > 0.001).float()
        scores = (one_hots * labels)
        return scores

    def validate(self, use_cuda):
        score = 0
        num_data = 0
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            if use_cuda:
                img = batch_data['img'].cuda()
                question = batch_data['question'].cuda()
                target = batch_data['target'].cuda()
            else:
                img = batch_data['img']
                question = batch_data['question']
                target = batch_data['target']
            pred = self._model(img, question)
            batch_score = self._compute_score_with_logits(pred, target).sum()
            score += batch_score
            num_data += pred.size(0)

        score = score / num_data
        return score


    def train(self, use_cuda):
        write = SummaryWriter(comment='_{}_loss_acc'.format(self._runner_name))
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                if use_cuda:
                    img = batch_data['img'].cuda()
                    question = batch_data['question'].cuda()
                    target = batch_data['target'].cuda()
                else:
                    img = batch_data['img']
                    question = batch_data['question']
                    target = batch_data['target']
                # print('\n Before run model in batch {}'.format(batch_id))
                pre = self._model(img, question)
                # print('\n After run model in batch {}'.format(batch_id))
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                predicted_answer = pre
                ground_truth_answer = target
                # ============

                # Optimize the model according to the predictions
                # print('\n Before optimize in batch {}'.format(batch_id))
                loss = self._optimize(predicted_answer, ground_truth_answer)
                # print('\n After optimize in batch {}'.format(batch_id))

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    write.add_scalar('{}/loss'.format(self._runner_name), loss, current_step)

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(use_cuda=use_cuda)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    write.add_scalar('{}/acc'.format(self._runner_name), val_accuracy, current_step)
