from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from models import Generator , G_loss , Discriminator , D_loss
from utils import load_dataset_mnist
from train import train



if __name__=='__main__':

    code_parser = argparse.ArgumentParser()
    code_parser.add_argument('--g_lr',metavar='g_lr',default='1e-4',type=float)
    code_parser.add_argument('--d_lr',metavar='d_lr',default='1e-4',type=float)
    code_parser.add_argument('--epoches',metavar='epoches',default='1',type=int)
    code_parser.add_argument('--g_batch_size',metavar='g_batch_size',default='16',type=int)
    code_parser.add_argument('--noise_dim',metavar='noise_dim',default='100',type=int)

    
    
    # for test
    '''
    args = code_parser.parse_args(
            '--g_lr=1e-4 --d_lr=1e-4 --epoches=1 --g_batch_size=16'.split())
    '''
    
    args = code_parser.parse_args()

        
    img_shape = (28,28,1)
    buffer_size = 60000
    batches = 256
    train_dataset = load_dataset_mnist(img_shape , buffer_size , batches)
    
    
    train(train_dataset,
             g_lr = args.g_lr,
             d_lr = args.d_lr,
             epoches = args.epoches,
             g_batch_size = args.g_batch_size,
             noise_dim = args.noise_dim)