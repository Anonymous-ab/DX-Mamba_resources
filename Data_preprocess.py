import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
import numpy as np
import pandas as pd
import re
from scipy.io import savemat
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type = str, default = 'Dog-X-ray', help= 'the name of the dataset') # HLC for human lung dataset
parser.add_argument('--input_captions_path', type = str, default = '/scratch/YoushanZhang/Dog_report/Report_generation', help = 'input captions json file') # '/scratch/YoushanZhang/Human_lung' for HLC
parser.add_argument('--save_dir', type = str, default = './data/Dog-X-ray/')
parser.add_argument('--word_count_threshold', default=5, type=int)

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}

def main(args):

    input_captions_path = args.input_captions_path
    input_vocab_json = ''
    output_vocab_json = 'vocab.json'
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Loading captions')
    # assert args.dataset in {'HLC'}




    if args.dataset == 'Dog-X-ray':

        # File paths for the captions
        train_captions = input_captions_path + '/Train.csv'
        valid_captions = input_captions_path + '/Valid.csv'
        test_captions =  input_captions_path + '/Test.csv'   
        train_data = pd.read_csv(train_captions)
        valid_data = pd.read_csv(valid_captions)
        test_data = pd.read_csv(test_captions)
        print(len(train_data))
        print(len(valid_data))
        print(len(test_data))
        data = pd.concat([train_data, valid_data, test_data], axis=0)

        print(data)
        print(len(data))
        print(data.tail(10))
        # Read image paths and captions for each image
        max_length = -1
        all_cap_tokens = []
        tokens_list = []

        # df.iloc[:, 6] = df.iloc[:, 6] + 'Can you generate the report based on these dog X-ray radiographs? '

        # data['Tokenized_Caption'] = "String"
        # Ques = 'Can you generate the report based on these human lung X-ray radiographs?'
        # Ques_tokens = re.findall(r'\w+|[^\w\s]', Ques)
        # print(Ques_tokens)
        # print(Ques + ' ' + data.iloc[0, 1] )
        # print(xx)
        # data.iloc[0, 1]  = Ques + ' ' + data.iloc[0, 1]  # Add questions to one caption to save possible words
        for i in range(len(data)):

            if str(data.iloc[i, 6])=="nan":
                questions =  'Can you generate the report based on these dog X-ray radiographs? '
            else:
                questions = str(data.iloc[i, 6]) + ' Can you generate the report based on these dog X-ray radiographs? '

            cap = questions + data.iloc[i, 7] 
            # print(cap)
            # print(uu)
            cap_tokens =  re.findall(r'\w+|[^\w\s]', cap)
            cap_tokens.insert(0, '<START>')  # Add at the beginning
            cap_tokens.append('<END>')        # Add at the end

            tokens_list.append(cap_tokens)
            max_length = max(max_length, len(cap_tokens))


        # all_cap_tokens.append((img['filename'], tokens_list))  
        print(data)
        print(data.tail(10))
        print(data.iloc[1000, 4])
        print(data.iloc[1000, 1])
        print(max_length)

    elif args.dataset == 'HLC':
        # File paths for the captions
        train_captions = input_captions_path + '/Train_captions.csv'
        valid_captions = input_captions_path + '/Valid_captions.csv'
        test_captions =  input_captions_path + '/Test_captions.csv'   
        train_data = pd.read_csv(train_captions)
        valid_data = pd.read_csv(valid_captions)
        test_data = pd.read_csv(test_captions)
        print(len(train_data))
        print(len(valid_data))
        print(len(test_data))
        data = pd.concat([train_data, valid_data, test_data], axis=0)

        print(data)
        print(len(data))
        print(data.tail(10))
        # Read image paths and captions for each image
        max_length = -1
        all_cap_tokens = []
        tokens_list = []
        data['Tokenized_Caption'] = "String"
        Ques = 'Can you generate the report based on these human lung X-ray radiographs?'
        Ques_tokens = re.findall(r'\w+|[^\w\s]', Ques)
        print(Ques_tokens)
        print(Ques + ' ' + data.iloc[0, 1] )
        # print(xx)
        data.iloc[0, 1]  = Ques + ' ' + data.iloc[0, 1]  # Add questions to one caption to save possible words
        for i in range(len(data)):
            cap = data.iloc[i, 1]
            # print(cap)
            # print(oo)
            # cap_tokens = tokenize(cap,
            #                     add_start_token=True,
            #                     add_end_token=True,
            #                     punct_to_keep=[';', ','],
            #                     punct_to_remove=['?', '.'])            
            # cap_tokens = tokenize(cap,
            #                     add_start_token=True,
            #                     add_end_token=True,
            #                     punct_to_keep=[';', ',', '?', '.', '-'])
            cap_tokens =  re.findall(r'\w+|[^\w\s]', cap)
            cap_tokens.insert(0, '<START>')  # Add at the beginning
            cap_tokens.append('<END>')        # Add at the end

            tokens_list.append(cap_tokens)
            max_length = max(max_length, len(cap_tokens))
            data.iloc[i, 4] = ' '.join(cap_tokens)

        # all_cap_tokens.append((img['filename'], tokens_list))  
        print(data)
        print(data.tail(10))
        print(data.iloc[1000, 4])
        print(data.iloc[1000, 1])
        print(max_length)


    # print(tokens_list[0])
    # print(p)
    print('max_length of the dataset:', max_length)
    # Either create the vocab or load it from disk
    if input_vocab_json == '':
        print('Building vocab')
        word_freq = build_vocab(tokens_list, 1)
    else:
        print('Loading vocab')
        with open(input_vocab_json, 'r') as f:
            word_freq = json.load(f)
    if output_vocab_json != '':
        with open(os.path.join(save_dir + output_vocab_json), 'w') as f:
            json.dump(word_freq, f)
    print('Finished vocab')


    if args.dataset == 'HLC':
    # Save the same question as arrary that we do not need to process it again during the data loader
        max_length = 270
        token_ques = np.zeros((max_length),dtype=int)
        # token_all_len = np.zeros((len(caption_list),1),dtype=int)

        tokens_encode = encode(Ques_tokens, word_freq,
                            allow_unk=0)
        token_ques[:len(tokens_encode)] = tokens_encode
        print(token_ques)
        savemat(os.path.join(save_dir +'question_token.mat'), {'token':token_ques})

        pred_caption = ""
        for i in token_ques:
            pred_caption += (list(word_freq.keys())[i]) + " "
        print(pred_caption)

    if args.dataset == 'Dog-X-ray': ### Test an example 
    # Save the same question as arrary that we do not need to process it again during the data loader
        max_length = 950
        token_ques = np.zeros((max_length),dtype=int)
        # token_all_len = np.zeros((len(caption_list),1),dtype=int)
        if str(data.iloc[9999, 6])=="nan":
            questions =  'Can you generate the report based on these dog X-ray radiographs? '
        else:
            questions = str(data.iloc[i, 6]) + ' Can you generate the report based on these dog X-ray radiographs? '

        cap = questions + data.iloc[i, 7] 

        cap_tokens =  re.findall(r'\w+|[^\w\s]', cap)
        cap_tokens.insert(0, '<START>')  # Add at the beginning
        cap_tokens.append('<END>')        # Add at the end

        tokens_encode = encode(cap_tokens, word_freq,
                            allow_unk=0)
        token_ques[:len(tokens_encode)] = tokens_encode
        print(token_ques)
        # savemat(os.path.join(save_dir +'question_token.mat'), {'token':token_ques})

        pred_caption = ""
        for i in token_ques:
            pred_caption += (list(word_freq.keys())[i]) + " "
        print(pred_caption)


def tokenize(s, delim=' ',add_start_token=True, 
    add_end_token=True, punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    for q in tokens:
        if q == '':
            tokens.remove(q)
    if tokens[0] == '':
        tokens.remove(tokens[0])
    if tokens[-1] == '':
        tokens.remove(tokens[-1])
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, min_token_count=1):#Calculate the number of independent words and tokenize vocab
    token_to_count = {}
    # for it in sequences:
    #     for seq in it[1]:
    # print(sequences)
    for seq in sequences:
        # print(seq)
        for token in seq:
            # print(token)
            # print(cc)
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1
    # print(token_to_count)
    # print(y)
    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

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

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
