import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from student_code.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from student_code.coattention_experiment_runner import CoattentionNetExperimentRunner
from student_code.custom_experiment_runner import CustomNetExperimentRunner
import torch

IMG_ROOT = '/media/iecas/Disk2/dataset/ms_coco'
DATA_ROOT = 'data/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--model', type=str, choices=['simple', 'coattention', 'custom'], default='custom')
    parser.add_argument('--train_image_dir', type=str, default=os.path.join(IMG_ROOT, 'train2014'))
    parser.add_argument('--train_question_path', type=str,
                        default=os.path.join(DATA_ROOT, 'OpenEnded_mscoco_train2014_questions.json'))
    parser.add_argument('--train_annotation_path', type=str,
                        default=os.path.join(DATA_ROOT, 'mscoco_train2014_annotations.json'))
    parser.add_argument('--test_image_dir', type=str, default=os.path.join(IMG_ROOT, 'val2014'))
    parser.add_argument('--test_question_path', type=str,
                        default=os.path.join(DATA_ROOT, 'OpenEnded_mscoco_val2014_questions.json'))
    parser.add_argument('--test_annotation_path', type=str,
                        default=os.path.join(DATA_ROOT, 'mscoco_val2014_annotations.json'))
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.debug:
        args.train_question_path = os.path.join(DATA_ROOT, 'test_questions.json')
        args.test_question_path = os.path.join(DATA_ROOT, 'test_questions.json')
        args.train_annotation_path = os.path.join(DATA_ROOT, 'test_annotations.json')
        args.test_annotation_path = os.path.join(DATA_ROOT, 'test_annotations.json')
        args.train_image_dir = DATA_ROOT
        args.test_image_dir = DATA_ROOT

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    elif args.model == "custom":
        experiment_runner_class = CustomNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                args=args,)
    experiment_runner.train(use_cuda=args.use_cuda)
