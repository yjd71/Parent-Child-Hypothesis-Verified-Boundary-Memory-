import argparse

def get_train_parser():
    parser = argparse.ArgumentParser(description='TalNet Args')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--config', default='config/runs/semisp_train.py', type=str)
    return parser

def get_tool_parser():
    parser = argparse.ArgumentParser(description='TalNet Tool Args')
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save pseudo labels')
    parser.add_argument('--split', default=None, type=float, help='data split')
    return parser

def get_eval_parser():
    parser = argparse.ArgumentParser(description='TalNet Eval Args')
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str, help='path to checkpoint')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save predictions')
    parser.add_argument('--testset', default=None, type=str, help='test set name')
    return parser
