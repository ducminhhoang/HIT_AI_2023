import os
from test import network_test
import argparse

'''
ARGUMENT: 
--cuda_device_no
--model_load_path
--test_content
--imsize
--output

'''
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-device-no', type=int,
                    help='cpu: -1, cuda: 0 ~ n', default=-1)
    parser.add_argument('--model_load_path', type=str,
                        help='Trained model load path')
    parser.add_argument('--test_content', type=str,
                        help='test content image path')
    parser.add_argument('--imsize', type=int,
                        help='Size for resize image during testing', default=256)
    parser.add_argument('--output', type=str,
                        help='output image path to save the stylized image')
    return parser

parser = build_parser()
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)
network_test(args)
