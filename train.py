import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="3"; 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json, random
from tqdm import tqdm
# from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from data.HLC.Cap_dataloader import Caption_Dataset
# from model.model_encoder_attMamba import Encoder, AttentiveEncoder
from model.model_encoder_Triple_Att_Mamba import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
import torch 

## Running code example
# python train.py --network MambaVision-L-1K --train_batchsize 32

def intra_dis(pred, labels):
    num_classes = labels.max() + 1
    # print(num_classes.cuda())
    # print(pred)
    # print(pred.size(1))
    # print(pred.size(1)).scatter_add(0, labels.view(-1, 1).expand(-1, pred.size(1)).cuda())
    class_sums = torch.zeros(num_classes.cuda(), pred.size(1)).cuda().scatter_add(0, labels.view(-1, 1).expand(-1, pred.size(1)).cuda(), pred)

    # Step 2: Calculate the count of samples per class (no loop)
    class_counts = torch.bincount(labels, minlength=num_classes).view(-1, 1).cuda()

    # Step 3: Calculate class centers by dividing sum by count
    class_centers = class_sums / class_counts

    # Step 4: Calculate distances to class centers (no loop)
    distances_to_center = torch.norm(pred - class_centers[labels], dim=1)
    all_dis = sum(distances_to_center)
    # print("Distances to respective class centers:\n", distances_to_center)
    # print("All Distances to respective class centers:\n", sum(distances_to_center))
    return all_dis

def inter_dis(pred, labels):
    # Step 1: Separate predictions by class and compute class centers
    unique_labels = labels.unique()  # Get unique classes
    class_centers = torch.stack([pred[labels == label].mean(dim=0) for label in unique_labels]).cuda()

    # Step 2: Compute all pairwise distances between class centers in one operation
    # Expanding dimensions to allow broadcasting for pairwise distance calculation
    diffs = class_centers[:, None, :] - class_centers[None, :, :]
    distances = torch.norm(diffs.cuda(), dim=2)  # Euclidean distance along the last dimension

    # print("Pairwise distances between class centers:\n", distances)


    # Mask to select only the elements strictly below the diagonal
    mask = torch.tril(torch.ones_like(distances), diagonal=-1).cuda()  # Creates a lower triangle mask without the diagonal

    # Apply the mask and sum
    sum_below_diagonal = torch.sum(distances * mask)

    # print("Sum of elements strictly below the diagonal:", sum_below_diagonal.item())
    return sum_below_diagonal.item()

class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.args = args
        random_str = str(random.randint(1, 10000))
        name = args.decoder_type+f'layers{args.decoder_n_layers}'+ time_file_str() +'_' +random_str
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath)==False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.04  # BLEU-4 score right now
        self.start_epoch = 0
        # with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        #     self.word_vocab = json.load(f)
        if args.data_name == 'Dog-X-ray':
            with open('./data/Dog-X-ray/vocab.json', 'r') as f:
                self.word_vocab = json.load(f)
        elif args.data_name == 'HLC':
            with open('./data/HLC/vocab.json', 'r') as f:
                self.word_vocab = json.load(f)
        # Initialize / load checkpoint
        self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_cap_cls = torch.nn.CrossEntropyLoss().cuda()

        # Custom dataloaders
        if args.data_name == 'Dog-X-ray':
            self.train_loader = data.DataLoader(
                Caption_Dataset(args.network,args.data_folder + '/Train_Images/', args.caption_path + './Train.csv', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                Caption_Dataset(args.network,args.data_folder + '/Valid_Images/', args.caption_path + './Valid.csv', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

            self.test_loader = data.DataLoader(
                Caption_Dataset(args.network,args.data_folder + '/Test_Images/', args.caption_path + './Test.csv', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        if args.data_name == 'HLC':
            self.train_loader = data.DataLoader(
                Caption_Dataset(args.network,args.data_folder + '/Train/', args.caption_path + './Train.csv', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                Caption_Dataset(args.network,args.data_folder + '/Valid/', args.caption_path + './Valid.csv', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)

            self.test_loader = data.DataLoader(
                Caption_Dataset(args.network,args.data_folder + '/Test/', args.caption_path + './Test.csv', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)



        self.l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
        self.l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 5))
        # Epochs

        self.best_model_path = None
        self.best_epoch = 0

    def build_model(self):
        args = self.args
        # Initialize / load checkpoint
        self.encoder = Encoder(args.network)
        self.encoder.fine_tune(args.fine_tune_encoder)
        self.encoder_trans = AttentiveEncoder(n_layers=args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                              heads=args.n_heads, max_length = args.max_length,  vocab_size=len(self.word_vocab), dropout=args.dropout)
        self.decoder = DecoderTransformer(decoder_type=args.decoder_type,
                                          embed_dim=args.embed_dim,
                                          vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                          word_vocab=self.word_vocab, n_head=args.n_heads,
                                          n_layers=args.decoder_n_layers, dropout=args.dropout)

        # set optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr)
        # checkpoint = torch.load('./models_ckpt/transformer_decoderlayers12024-10-31-12-19-11_4672/Dog-X-ray_bts_64_MambaVision-L-1K_epo_88_Bleu4_24302.pth')
        checkpoint = torch.load('./models_ckpt/transformer_decoderlayers12024-11-03-16-18-53_1327/Dog-X-ray_bts_32_MambaVision-L-1K_epo_89_Bleu4_24980_test.pth') #### The best

        # Move to GPU, if available
        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()
        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_dict'])

        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0) if args.fine_tune_encoder else None
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,
                                                                    gamma=1.0)

    def training(self, args, epoch):
        self.encoder.train()
        self.encoder_trans.train()
        self.decoder.train()  # train mode (dropout and batchnorm is used)

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        for id, batch_data in enumerate(self.train_loader):
            # if id == 10:
            #    break
            start_time = time.time()
            accum_steps = 64//args.train_batchsize

            # Move to GPU, if available
            imgA = batch_data['imgA']
            imgB = batch_data['imgB']
            imgC = batch_data['imgC']
            # token = batch_data['token']
            # token_len = batch_data['token_len']
            token = batch_data['token_all']
            token_len = batch_data['token_all_len']
            Ques = batch_data['Ques'].squeeze(1)
            # print(imgA.shape)
            # # print(imgB.shape)
            # print(token.shape)
            # print(token_len)
            # # print(batch_data['token_all'].shape)
            # # print(batch_data['token_all_len'].shape)
            # print(dd)
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            imgC = imgC.cuda()
            token = token.cuda()
            token_len = token_len.cuda()
            Ques = Ques.cuda()
            # Forward prop.
            feat1, feat2, feat3 = self.encoder(imgA, imgB, imgC)
            # print(Ques.shape)
            # print(ddd)
            feat = self.encoder_trans(feat1, feat2, feat3, Ques)
            # print(feat.shape)
            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat, token, token_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # Calculate loss
            loss = self.criterion_cap(scores, targets.to(torch.int64))
            # print(scores.shape)
            # print(targets.shape)
            all_dis = intra_dis(scores, targets.to(torch.int64))
            # print('ccc',all_dis/100000)
            each_dis = inter_dis(scores, targets.to(torch.int64)) 
            # print(1.0/each_dis*1e8)
            # print(iio)
            # Back prop.
            loss = loss / accum_steps + all_dis/100000 + 1.0/each_dis*1e8
            loss.backward()

            # self.encoder_optimizer.zero_grad() 
            # self.decoder_optimizer.zero_grad()
            # self.encoder_trans_optimizer.zero_grad()
            
            # self.encoder_optimizer.step()   
            # self.decoder_optimizer.step()
            # self.encoder_trans_optimizer.step()

            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)

            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()

                # Adjust learning rate
                self.decoder_lr_scheduler.step()
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    self.encoder_lr_scheduler.step()

                self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time #batch_time
            self.hist[self.index_i, 1] = loss.item()  # train_loss
            self.hist[self.index_i, 2] = accuracy_v0(scores, targets, 3) #top5

            self.index_i += 1
            # Print status
            if self.index_i % args.print_freq == 0:
                print_log('Training Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Cap_loss: {4:.5f}\t'
                    'Text_Top-5 Acc: {5:.3f}'
                    .format(epoch, id, len(self.train_loader),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,0])*args.print_freq,
                                         np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,1]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,2])
                                ), self.log)

    # One epoch's validation
    def validation(self, epoch, data):
        word_vocab = self.word_vocab
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        with torch.no_grad():
            # Batches
            if data ==  "Valid":
                dataset = self.val_loader
            else:
                dataset = self.test_loader

            for ind, batch_data in enumerate(
                    tqdm(dataset, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                # if ind == 20:
                #     break
                # Move to GPU, if available
                # (imgA, imgB, token_all, token_all_len, _, _, _)
                imgA = batch_data['imgA']
                imgB = batch_data['imgB']
                imgC = batch_data['imgC']
                token_all = batch_data['token_all']
                token_all_len = batch_data['token_all_len']
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                imgC = imgC.cuda()
                token_all = token_all.squeeze(0).cuda()
                Ques = batch_data['Ques'].squeeze(1)
                Ques = Ques.cuda()
                # Forward prop.
                if self.encoder is not None:
                    feat1, feat2, feat3 = self.encoder(imgA, imgB, imgC)
                feat = self.encoder_trans(feat1, feat2, feat3, Ques)
                # print(feat.shape)
                seq = self.decoder.sample(feat, k=1)

                # for captioning
                except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
                img_token = token_all.tolist()
                # print(img_token)
                img_tokens = [w for w in img_token if w not in except_tokens]
                # img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],
                #         img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in except_tokens]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % self.args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        # for j in i:
                        ref_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption += ".    "
                # if ind ==3:
                #     print('cc'*50)
                #     print(pred_caption)
                #     print('xx'*50)
                #     print(ref_caption)
                #     print(references)
                #     print(hypotheses)
                #     break;
            val_time = time.time() - val_start_time
            print(pred_caption)
            print(ref_caption)
            # Fast test during the training
            # Calculate evaluation scores

            # print('uu'*50)
            print(len(references))
            print(len(hypotheses))
            # print(pp)

            score_dict = get_eval_score(references, hypotheses)
            # print(score_dict)
            # print(pp)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print_log('Captioning_Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), self.log)

        # Check if there was an improvement
        if Bleu_4 > self.best_bleu4:
            self.best_bleu4 = max(Bleu_4, self.best_bleu4)
            # save_checkpoint
            print('Save Model')
            state = {'encoder_dict': self.encoder.state_dict(),
                     'encoder_trans_dict': self.encoder_trans.state_dict(),
                     'decoder_dict': self.decoder.state_dict()
                     }
            metric = f'Bleu4_{round(100000 * self.best_bleu4)}'
            model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
            if epoch > 4:
                torch.save(state, os.path.join(self.args.savepath, model_name.replace('/','-')))
            # save a txt file
            text_path = os.path.join(self.args.savepath, model_name.replace('/','-'))
            with open(text_path.replace('.pth', '.txt'), 'w') as f:
                f.write('Bleu_1: ' + str(Bleu_1) + '\t')
                f.write('Bleu_2: ' + str(Bleu_2) + '\t')
                f.write('Bleu_3: ' + str(Bleu_3) + '\t')
                f.write('Bleu_4: ' + str(Bleu_4) + '\t')
                f.write('Meteor: ' + str(Meteor) + '\t')
                f.write('Rouge: ' + str(Rouge) + '\t')
                f.write('Cider: ' + str(Cider) + '\t')

        if (data ==  "Test"):
            print('Save Model')
            state = {'encoder_dict': self.encoder.state_dict(),
                    'encoder_trans_dict': self.encoder_trans.state_dict(),
                    'decoder_dict': self.decoder.state_dict()
                    } 
            metric = f'Bleu4_{round(100000 * Bleu_4)}'               
            model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}_test.pth'
            torch.save(state, os.path.join(self.args.savepath, model_name.replace('/','-')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument('--sys', default='win', choices=('linux'), help='system')
    parser.add_argument('--data_folder', default='/scratch/YoushanZhang/Dog_report/Report_generation',help='folder with data files') # '/scratch/YoushanZhang/Human_lung/Images' for HLC
    parser.add_argument('--caption_path', default='/scratch/YoushanZhang/Dog_report/Report_generation/', help='path of the data lists') # '/scratch/YoushanZhang/Human_lung/' for HLC
    parser.add_argument('--token_folder', default='./data/Dog-X-ray/', help='folder with token files') # './data/HLC/' for HLC
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=950, help='path of the data lists') # 270 for HLC
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="Dog-X-ray",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=16, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_ckpt/")
    # backbone parameters
    parser.add_argument('--network', default='CLIP-ViT-B/32', help=' define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
    parser.add_argument('--feat_size', type=int, default=7, help='size of extracted features of backbone')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
    args = parser.parse_args()


    if args.network == 'CLIP-RN50':
        clip_emb_dim = 1024
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN101':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN50x4':
        clip_emb_dim = 640
        args.encoder_dim, args.feat_size = 2560, 9
    elif args.network == 'CLIP-RN50x16':
        clip_emb_dim = 768
        args.encoder_dim, args.feat_size = 3072, 12
    elif args.network == 'CLIP-ViT-B/16' or args.network == 'CLIP-ViT-L/16':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 14
    elif args.network == 'CLIP-ViT-B/32' or args.network == 'CLIP-ViT-L/32':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 7
    elif args.network == 'segformer-mit_b1':
        args.encoder_dim, args.feat_size = 512, 8

    args.embed_dim = args.encoder_dim

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
        trainer.training(trainer.args, epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        trainer.validation(epoch, data = "Valid")
        if (epoch+1) % 10 == 0:
            trainer.validation(epoch, data = "Test")

