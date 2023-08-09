import argparse
import os
import torch

from train import network_train
torch.cuda.empty_cache()
import gc

gc.collect()
# torch.cuda.memory_summary(device=None, abbreviated=False)

'''
ARGUMENT: Syntax : -- + name_of_argument, use: args.name_of_argument
--cuda_device_no
--train_content
--imsize
--cropsize
--lr
--vgg_flag
--max_iter
--iter_check
--batchs
--train_style
--style_layers
--content_layers
--content_weight
--style_weight
--tv_weight
--save_path
'''

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device_no', type=int,
                        help='cpu: -1, gpu: 0 ~ n', default=-1)
    
    parser.add_argument('--max_iter', type=int,
                        help='Train iterations', default=15000)
    
    parser.add_argument('--check_iter', type=int,
                        help='Number of iteration to check training logs', default=100)
    
    parser.add_argument('--batchs', type=int,
                        help='Batch size', default=8)
    parser.add_argument('--lr', type=int,
                        help='Learning rate to optimize network', default=0.1)
    
    parser.add_argument('--imsize', type=int,
                        help='Size of crop image during training', default=256)
    
    parser.add_argument('--cropsize', type=int,
                        help='Size for crop image durning training', default=240)
    
    parser.add_argument('--vgg_flag', type=str,
                        help='VGG flag for caculating losses', default='vgg16')
    
    parser.add_argument('--content_layers', type=int, nargs='+',
                        help='layer index to extract content features', default=[15])
    
    parser.add_argument('--style_layers', type=int, nargs='+',
                        help='layer index to extract style features', default=[3, 8, 15, 22])
    
    parser.add_argument('--content-weight', type=float, 
                    help='content loss weight', default=1.0)
    
    parser.add_argument('--style-weight', type=float,
                    help='style loss weight', default=30.0)
    
    parser.add_argument('--tv-weight', type=float,
                    help='tv loss weight', default=1.0)
    parser.add_argument('--train_content', type=str,
                        help='Content images path for training')
    
    parser.add_argument('--train_style', type=str,
                        help='Style target image path for trainging')
    
    parser.add_argument('--save_path', type=str,
                        help='Save path')

    return parser

parser = build_parser()
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_no)
transform_network = network_train(args)
print('Done')
