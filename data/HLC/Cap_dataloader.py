import clip
import torch
from torch.utils.data import Dataset
# from preprocess_data import encode
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
#import cv2 as cv
from imageio import imread
from random import *
from PIL import Image
import skimage.io as io
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import re

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

class Caption_Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, encoder_network, data_folder, caption_path, token_folder = None, word_vocab = None, max_length = 270, allow_unk = 0, max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param caption_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.encoder_network = encoder_network
        if 'CLIP' in self.encoder_network:
            encoder_network = encoder_network.replace('CLIP-','')
            clip_model, self.preprocess = clip.load(encoder_network, device='cpu', jit=False)

        # self.mean=[100.6790,  99.5023,  84.9932]
        # self.std=[50.9820, 48.4838, 44.7057]
        self.mean = [0.39073*255,  0.38623*255, 0.32989*255]
        self.std = [0.15329*255,  0.14628*255, 0.13648*255]
        self.caption_path = caption_path
        self.max_length = max_length

        self.word_vocab = word_vocab
        self.allow_unk = allow_unk
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.data = pd.read_csv(caption_path)
        self.data_folder = data_folder
        # Load the .mat file
        if "Dog" in self.caption_path:
            self.set = "Dog"
        elif "Human" in self.caption_path:
            self.set = "Human"
            token = loadmat('./data/HLC/question_token.mat')
            self.ques = token['token']


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.set == "Dog":
            image_path1 = os.path.join(self.data_folder,self.data.iloc[idx, 0])
            image_path2 = os.path.join(self.data_folder, self.data.iloc[idx, 1])
            image_path3 = os.path.join(self.data_folder, self.data.iloc[idx, 2])

            if str(self.data.iloc[idx, 6])=="nan":
                questions =  'Can you generate the report based on these dog X-ray radiographs?'
            else:
                questions = str(self.data.iloc[idx, 6]) + ' Can you generate the report based on these dog X-ray radiographs?'

            Que_tokens =  re.findall(r'\w+|[^\w\s]', questions)
            Que_tokens.insert(0, '<START>')  # Add at the beginning
            Que_tokens.append('<END>')        # Add at the end

            Ques = np.zeros((self.max_length),dtype=int)
            # token_all_len = np.zeros((len(caption_list),1),dtype=int)

            tokens_que_encode = encode(Que_tokens, self.word_vocab,
                                allow_unk=self.allow_unk == 1)
            Ques[:len(tokens_que_encode)] = tokens_que_encode

            cap = self.data.iloc[idx, 7]

        elif self.set == "Human":        
            image_path1 = os.path.join(self.data_folder,self.data.iloc[idx, 2])
            image_path2 = os.path.join(self.data_folder, self.data.iloc[idx, 3])
            image_path3 = os.path.join(self.data_folder, self.data.iloc[idx, 2])
            Ques = self.ques
            cap = self.data.iloc[idx, 1]

        
        tokens =  re.findall(r'\w+|[^\w\s]', cap)
        tokens.insert(0, '<START>')  # Add at the beginning
        tokens.append('<END>')        # Add at the end

        token_all = np.zeros((self.max_length),dtype=int)
        # token_all_len = np.zeros((len(caption_list),1),dtype=int)

        tokens_encode = encode(tokens, self.word_vocab,
                            allow_unk=self.allow_unk == 1)
        token_all[:len(tokens_encode)] = tokens_encode
        token_all_len = len(tokens_encode)

        if 'CLIP' not in self.encoder_network:
            imgA = Image.fromarray(imread(image_path1)).resize((224, 224)).convert("RGB")
            imgB = Image.fromarray(imread(image_path2)).resize((224, 224)).convert("RGB")
            imgC = Image.fromarray(imread(image_path3)).resize((224, 224)).convert("RGB")
            # print('xx', np.asarray(imgA, np.float32).shape)
            imgA = np.asarray(imgA, np.float32).transpose(2, 0, 1)
            # print('cc', imgA.shape)
            imgB = np.asarray(imgB, np.float32).transpose(2, 0, 1)
            imgC = np.asarray(imgC, np.float32).transpose(2, 0, 1)
            for i in range(len(self.mean)):
                imgA[i, :, :] -= self.mean[i]
                imgA[i, :, :] /= self.std[i]
                imgB[i, :, :] -= self.mean[i]
                imgB[i, :, :] /= self.std[i]
                imgC[i, :, :] -= self.mean[i]
                imgC[i, :, :] /= self.std[i]                
            # print('tt',imgA.transpose(1, 2, 0).shape)
            # imgA = Image.fromarray(imgA.transpose(1, 2, 0)) #.resize((224, 224))
            # imgB = Image.fromarray(imgB.transpose(1, 2, 0)) #.resize((224, 224))
            # print(yyy)
        else:
            imgA = io.imread(image_path1)
            imgB = io.imread(image_path2)
            imgC = io.imread(image_path3)
            # print(imgA.shape)
            # CLIP process
            imgA = self.preprocess(Image.fromarray(imgA))
            imgB = self.preprocess(Image.fromarray(imgB))
            imgC = self.preprocess(Image.fromarray(imgC))
            # print(imgA.shape)
            # print(xx)


        # imgA, imgB, token_all, token_all_len, token, np.array(token_len), name
        out_dict = {
            'imgA': imgA,
            'imgB': imgB,
            'imgC': imgC,
            'token_all': token_all,
            'token_all_len': token_all_len,
            'Ques':Ques
        }

        # print(imgA.shape)
        # print(imgB.shape)
        # print(token_all.shape)
        # print(token_all_len)
        # print(Ques.shape)
        # # # print(token)
        # # # print(token_len)
        # # # print(name)
        # print(iio)
        return out_dict

#################### For Dog X-ray dataset
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

#     # Data parameters
#     parser.add_argument('--sys', default='win', choices=('linux'), help='system')
#     parser.add_argument('--data_folder', default='/scratch/YoushanZhang/Dog_report/Report_generation',help='folder with data files') # '/scratch/YoushanZhang/Human_lung/Images' for HLC
#     parser.add_argument('--caption_path', default='/scratch/YoushanZhang/Dog_report/Report_generation/', help='path of the data lists') # '/scratch/YoushanZhang/Human_lung/' for HLC
#     parser.add_argument('--token_folder', default='./data/Dog-X-ray/', help='folder with token files') # './data/HLC/' for HLC
#     parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
#     parser.add_argument('--max_length', type=int, default=950, help='path of the data lists') # 270 for HLC
#     parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
#     parser.add_argument('--data_name', default="Dog-X-ray",help='base name shared by data files.')

#     parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
#     parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
#     parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
#     # Training parameters
#     parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
#     parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
#     parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for (if early stopping is not triggered).')
#     parser.add_argument('--workers', type=int, default=16, help='for data-loading; right now, only 0 works with h5pys in windows.')
#     parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
#     parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
#     parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
#     # Validation
#     parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
#     parser.add_argument('--savepath', default="./models_ckpt/")
#     # backbone parameters
#     parser.add_argument('--network', default='CLIP-ViT-B/32', help=' define the backbone encoder to extract features')
#     parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
#     parser.add_argument('--feat_size', type=int, default=7, help='size of extracted features of backbone')
#     # Model parameters
#     parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
#     parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
#     parser.add_argument('--decoder_n_layers', type=int, default=1)
#     parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
#     args = parser.parse_args()


# with open('./data/Dog-X-ray/vocab.json', 'r') as f:
#     word_vocab = json.load(f)
    
# train_loader = DataLoader(
#     Caption_Dataset(args.network,args.data_folder + '/Train_Images/', args.caption_path + './Train.csv', args.token_folder, word_vocab, args.max_length, args.allow_unk),
#     batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

# for id, batch_data in enumerate(train_loader):
#     print(id)
#     # print(batch_data)
#     print(batch_data['token_all'].shape)
#     print(batch_data['token_all_len'].shape)
#     print(batch_data['Ques'].shape)
#     print(xx)

#################### For HLC
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

#     # Data parameters
#     parser.add_argument('--sys', default='win', choices=('linux'), help='system')
#     parser.add_argument('--data_folder', default='/scratch/YoushanZhang/Human_lung/Images',help='folder with data files')
#     parser.add_argument('--caption_path', default='/scratch/YoushanZhang/Human_lung/', help='path of the data lists')
#     parser.add_argument('--token_folder', default='./data1/LEVIR_CC/tokens/', help='folder with token files')
#     parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
#     parser.add_argument('--max_length', type=int, default=270, help='path of the data lists')
#     parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
#     parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')

#     parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
#     parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
#     parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
#     # Training parameters
#     parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
#     parser.add_argument('--train_batchsize', type=int, default=2, help='batch_size for training')
#     parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train for (if early stopping is not triggered).')
#     parser.add_argument('--workers', type=int, default=16, help='for data-loading; right now, only 0 works with h5pys in windows.')
#     parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
#     parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
#     parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
#     parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
#     # Validation
#     parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
#     parser.add_argument('--savepath', default="./models_ckpt/")
#     # backbone parameters
#     parser.add_argument('--network', default='CLIP-ViT-B/32', help=' define the backbone encoder to extract features')
#     parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
#     parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
#     # Model parameters
#     parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
#     parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
#     parser.add_argument('--decoder_n_layers', type=int, default=1)
#     parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
#     args = parser.parse_args()


# with open('./data/HLC/vocab.json', 'r') as f:
#     word_vocab = json.load(f)
    
# train_loader = DataLoader(
#     Caption_Dataset(args.network,args.data_folder + '/Train/', args.caption_path + './Train.csv', args.token_folder, word_vocab, args.max_length, args.allow_unk),
#     batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

# for id, batch_data in enumerate(train_loader):
#     print(id)
#     # print(batch_data)
#     print(batch_data['token_all'].shape)
#     print(batch_data['token_all_len'].shape)
#     print(batch_data['Ques'].shape)
#     print(xx)
