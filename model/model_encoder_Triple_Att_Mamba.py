import torch
from torch import nn, einsum
import torchvision.models as models
from einops import rearrange
import clip
from transformers import MambaConfig
from model.mamba_block import CaMambaModel
# from mamba_block import CaMambaModel
import numpy as np
# from transformers.models.mamba.modeling_mamba import MambaRMSNorm
class Encoder(nn.Module):
    """
    Encoder.
    """ 
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            self.cnn = models.alexnet(pretrained=True)
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            self.cnn = models.vgg19(pretrained=True)  
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            self.cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            self.modules = list(self.cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            self.cnn = models.resnet18(pretrained=True)  
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            self.cnn = models.resnet34(pretrained=True)  
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            self.cnn = models.resnet50(pretrained=True)  
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            self.cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            self.cnn = models.resnet152(pretrained=True)  
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            self.cnn = models.resnext50_32x4d(pretrained=True)  
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            self.cnn = models.resnext101_32x8d(pretrained=True)  
            self.modules = list(self.cnn.children())[:-1]
        elif self.network=='vit_l_16':#1024,1/16H,1/16W [1024, 14, 14]
            self.cnn =  models.vit_l_16(weights='IMAGENET1K_SWAG_E2E_V1')  # ImageNet 1K Top 1 88.064
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='regnet_y_32gf':#3712,1/32H,1/32W [3712, 7, 7]
            self.cnn = models.regnet_y_32gf(weights='IMAGENET1K_SWAG_E2E_V1')   # ImageNet 1K Top 1 86.838
            self.modules = list(self.cnn.children())[:-2]
        elif self.network=='MambaVision-T-1K':#3712,1/32H,1/32W [640, 7, 7]
            from transformers import AutoModel
            self.cnn = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True) 
            out_avg_pool, features = self.cnn(inputs) # features[3]: [640, 7, 7] ImageNet 1K Top 1 82.3
        elif self.network=='MambaVision-L-1K':#3712,1/32H,1/32W [1568, 7, 7] 
            from transformers import AutoModel
            self.cnn = AutoModel.from_pretrained("nvidia/MambaVision-L-1K", trust_remote_code=True) 
            # out_avg_pool, features = self.cnn(inputs) # features[3]: [1568, 7, 7] ImageNet 1K Top 1 85.0
        elif 'CLIP' in self.network:
            clip_model_type = self.network.replace('CLIP-', '')
            self.clip_model, preprocess = clip.load(clip_model_type, jit=False)  #
            self.clip_model = self.clip_model.to(dtype=torch.float32)

        # self.cnn_list = nn.ModuleList(modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, imageA, imageB, imageC):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if "CLIP" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            img_C = imageC.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)
            clip_emb_C, img_feat_C = self.clip_model.encode_image(img_C)            
        else:
            # feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            # feat2 = self.cnn(imageB)
            feat1 = imageA
            feat2 = imageB
            feat3 = imageC
            # print(feat1.shape)
            # print(feat2.shape)
            # print(self.modules(imageA).shape)
            # feat1_list = []
            # feat2_list = []
            # cnn_list = list(self.cnn.children())
            if self.network=='MambaVision-L-1K':
                _, feat1 = self.cnn(feat1)
                _, feat2 = self.cnn(feat2)
                _, feat3 = self.cnn(feat3)
                feat1 = feat1[3]
                feat2 = feat2[3]
                feat3 = feat3[3]
            else:
                cnn_list = self.modules
                for module in cnn_list:
                    feat1 = module(feat1)
                    feat2 = module(feat2)
                    feat3 = module(feat3)
                # feat1_list.append(feat1)
                # feat2_list.append(feat2)
            # print(feat1.shape)
            # print(feat2.shape)    
            feat1 = feat1.view(feat1.shape[0], feat1.shape[1], -1)
            feat2 = feat2.view(feat2.shape[0], feat2.shape[1], -1)
            feat3 = feat3.view(feat3.shape[0], feat3.shape[1], -1)

            img_feat_A = feat1[:, -768:, :].permute(0, 2, 1)
            img_feat_B = feat2[:, -768:, :].permute(0, 2, 1)
            img_feat_C = feat3[:, -768:, :].permute(0, 2, 1)
            # print(feat1.shape)
            # print(feat2.shape)       
            # feat1_list = feat1_list[-4:]
            # feat2_list = feat2_list[-4:]
            # print(feat1_list)
            # print(len(feat1_list))
            # print(feat1_list[0].shape)
            # print(torch.stack(feat1_list).shape)
        return img_feat_A, img_feat_B, img_feat_C

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 3 through 4
        if 'CLIP' in self.network and fine_tune:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune last 2 trans and ln_post
            children_list = list(self.clip_model.visual.transformer.resblocks.children())[-6:]
            children_list.append(self.clip_model.visual.ln_post)
            for c in children_list:
                for p in c.parameters():
                    p.requires_grad = True
        elif 'CLIP' not in self.network and fine_tune and (self.network!='MambaVision-L-1K'):
            # for c in list(self.cnn.children())[:]:
            for c in self.modules[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune



class resblock(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, int(outchannel / 1), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), int(outchannel / 1), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return self.act(out)

class AlignmentLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pooled_size, num_heads):
        super(AlignmentLayer, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.pool = nn.AdaptiveAvgPool1d(pooled_size)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        # Explicit linear layers for Q, K, V projections, we need to calculate the Q, K, V for the attention layer
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image_features, text_indices):
        # Step 1: Use the embedding layer for text features
        # print(text_indices.shape)
        # print(text_indices)
        # print(text_indices.max().item())
        text_embeddings = self.embedding_layer(text_indices)  # Shape: (2, 240, 768)
        # print(text_embeddings)
        # print(text_embeddings.shape)
        # print(text_embeddings.permute(0, 2, 1))
        # Step 2: Apply Adaptive Average Pooling to reduce to [2, 49, 768]
        pooled_text_features = self.pool(text_embeddings.permute(0, 2, 1))  # Shape: (2, 768, 240)
        pooled_text_features = pooled_text_features.permute(0, 2, 1)  # Back to [2, 49, 768]

        # Step 3: Prepare for multi-head attention
        # We need to transpose the dimensions to [seq_len, batch, embed_dim]
        image_features = image_features.permute(1, 0, 2)  # Shape: (49, 2, 768)
        pooled_text_features = pooled_text_features.permute(1, 0, 2)  # Shape: (49, 2, 768) ## Considering using three images to multiple text features as QKV

        # Step 2: Explicitly project image and text features to Q, K, V, please try this one for the next step
        # Transpose for [seq_len, batch, embed_dim] format required by nn.MultiheadAttention
        query = self.query_projection(image_features)  # Shape: (49, 2, 768)
        key = self.key_projection(pooled_text_features)# Shape: (49, 2, 768)
        value = self.value_projection(pooled_text_features)  # Shape: (49, 2, 768)
        # print(query.shape)
        # print(value.shape)
        # print('cc', image_features.shape)
        # print('xx', pooled_text_features.shape)
        # print(ioo)
        # Step 3: Apply multi-head attention with explicit Q, K, V
        attn_output, attn_weights = self.multihead_attention(query, key, value)

        # Step 4: Apply multi-head attention
        # attn_output, attn_weights = self.multihead_attention(image_features, pooled_text_features, pooled_text_features)

        # Step 5: Transpose back to [batch, seq_len, embed_dim]
        aligned_features = attn_output.permute(1, 0, 2)  # Shape: (2, 49, 768)

        return aligned_features

class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, max_length, vocab_size, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.n_layers = n_layers
        self.channels = channels
        # position embedding
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        # Mamba
        config_1 = MambaConfig(num_hidden_layers=1, conv_kernel=3,hidden_size=channels)
        config_2 = MambaConfig(num_hidden_layers=1, conv_kernel=3,hidden_size=channels)
        self.CaMalayer_list = nn.ModuleList([])
        self.fuselayer_list = nn.ModuleList([])
        self.fuselayer_list_2 = nn.ModuleList([])
        self.linear_dif = nn.ModuleList([])
        self.linear_img1 = nn.ModuleList([])
        self.linear_img2 = nn.ModuleList([])
        self.Dyconv_img1_list = nn.ModuleList([])
        self.Dyconv_img2_list = nn.ModuleList([])
        embed_dim = channels
        self.Conv1_list = nn.ModuleList([])
        self.LN_list = nn.ModuleList([])
        self.num_embeddings = vocab_size ## maximum of vocabulary need to be changed, 240, 270, a bigger number 
        self.embedding_dim = channels ## embedding size: 768
        self.pooled_size = h_feat * w_feat ## image feature suze 7 * 7 = 49
        self.num_heads = 8  # Number of attention heads
        self.alignment_layer = AlignmentLayer(self.num_embeddings, self.embedding_dim, self.pooled_size, self.num_heads).cuda()
        for i in range(n_layers):
            self.CaMalayer_list.append(nn.ModuleList([
                CaMambaModel(config_1),
                CaMambaModel(config_1),
            ]))
            self.fuselayer_list.append(nn.ModuleList([
                CaMambaModel(config_2),
                CaMambaModel(config_2),
            ]))
            # self.linear_dif.append(nn.Sequential(
            #     nn.Linear(channels, channels),
            #     # nn.SiLU(),
            # ))
            # self.Dyconv_img1_list.append(Dynamic_conv(channels))
            # self.Dyconv_img2_list.append(Dynamic_conv(channels))
            # self.Dyconv_dif_list.append(Dynamic_conv(channels))
            # self.linear_img1.append(nn.Linear(2*channels, channels))
            # self.linear_img2.append(nn.Linear(2*channels, channels))
            self.Conv1_list.append(nn.Conv2d(channels * 3, embed_dim, kernel_size=1)) # 2 for two image features (768 *2) and 3 for 3 images (768 *3)
            self.LN_list.append(resblock(embed_dim, embed_dim))
        self.act = nn.Tanh()
        self.layerscan = CaMambaModel(config_1)
        self.LN_norm = nn.LayerNorm(channels)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Fusion bi-temporal feat for captioning decoder
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        if len(x.shape) == 3: # NLD
            b = x.shape[0]
            c = x.shape[-1]
            # print(x.transpose(-1, 1).shape)
            # print(self.h_feat)
            # print(self.w_feat)
            x = x.transpose(-1, 1).view(b, c, self.h_feat, self.w_feat)
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        # reshape back to NLD
        x = x.view(b, c, -1).transpose(-1, 1)  # NLD (b,hw,c)
        return x

    def forward(self, img_A, img_B, img_C, Ques):
        h, w = self.h_feat, self.w_feat
        # print("xx",img_A.shape)
        # print("xx",img_B.shape)
        # print("xx",img_C.shape)
        # 1. A B feature from backbone  NLD
        img_A = self.add_pos_embedding(img_A)
        img_B = self.add_pos_embedding(img_B)
        img_C = self.add_pos_embedding(img_C)
        # print("xx",img_A.shape)
        # captioning
        batch, c = img_A.shape[0], img_A.shape[-1]
        img_sa1, img_sa2, img_sa3 = img_A, img_B, img_C

        # Method: Mamba
        # self.CaMalayer_list.train()
        img_list = []
        N, L, D = img_sa1.shape
        for i in range(self.n_layers):
            # # SD-SSM:
            # dif = img_sa2 + img_sa1
            # img_sa1 = self.CaMalayer_list[i][0](inputs_embeds=img_sa1, inputs_embeds_2=dif).last_hidden_state
            # img_sa2 = self.CaMalayer_list[i][1](inputs_embeds=img_sa2, inputs_embeds_2=dif).last_hidden_state

            
            # Triple-ATT-SSM:
            # dif = img_sa2 + img_sa1
            # print(img_sa1.shape)
            # print(Ques.shape)
            # print(xx)
            aligned1 = self.alignment_layer(img_sa1, Ques)
            aligned2 = self.alignment_layer(img_sa2, Ques)
            aligned3 = self.alignment_layer(img_sa3, Ques)
            # print(aligned1.shape)
            # print(aligned2.shape)
            # img_sa1 = self.CaMalayer_list[i][0](inputs_embeds=img_sa1, inputs_embeds_2=aligned1).last_hidden_state
            # img_sa2 = self.CaMalayer_list[i][1](inputs_embeds=img_sa2, inputs_embeds_2=aligned2).last_hidden_state
            # img_sa3 = self.CaMalayer_list[i][1](inputs_embeds=img_sa3, inputs_embeds_2=aligned3).last_hidden_state            
            img_sa1 = self.CaMalayer_list[i][0](inputs_embeds=img_sa1, inputs_embeds_2=aligned1+aligned2+aligned3).last_hidden_state
            img_sa2 = self.CaMalayer_list[i][1](inputs_embeds=img_sa2, inputs_embeds_2=aligned2+aligned2+aligned3).last_hidden_state
            img_sa3 = self.CaMalayer_list[i][1](inputs_embeds=img_sa3, inputs_embeds_2=aligned3+aligned2+aligned3).last_hidden_state

            # TT-SSM:
            scan_mode = 'TT-SSM'
            if scan_mode == 'TT-SSM':
                img_sa1 = self.LN_norm(img_sa1)#
                img_sa2 = self.LN_norm(img_sa2)#
                img_sa3 = self.LN_norm(img_sa3)#
                img_sa1_res = img_sa1
                img_sa2_res = img_sa2
                img_sa3_res = img_sa3
                img_fuse1 = img_sa1.permute(0, 2, 1).unsqueeze(-1) # (N,D,L,1)
                img_fuse2 = img_sa2.permute(0, 2, 1).unsqueeze(-1)
                img_fuse3 = img_sa3.permute(0, 2, 1).unsqueeze(-1)

                img_fuse = torch.cat([img_fuse1, img_fuse2, img_fuse3], dim=-1).reshape(N, D, -1) # (N,D,L*3)
                img_fuse = self.fuselayer_list[i][0](inputs_embeds=img_fuse.permute(0, 2, 1)).last_hidden_state.permute(0, 2, 1) # (N,D,L*3)

                # img_fuse = torch.cat([img_fuse1, img_fuse2], dim=-1).reshape(N, D, -1) # (N,D,L*2)
                # img_fuse = self.fuselayer_list[i][0](inputs_embeds=img_fuse.permute(0, 2, 1)).last_hidden_state.permute(0, 2, 1) # (N,D,L*3)


                img_fuse = img_fuse.reshape(N, D, L, -1)

                img_sa1 = img_fuse[..., 0].permute(0, 2, 1)#[...,:D] # (N,L,D)
                img_sa2 = img_fuse[..., 1].permute(0, 2, 1)#[...,:D]
                img_sa3 = img_fuse[..., 2].permute(0, 2, 1)#[...,:D] # (N,L,D)
                #
                img_sa1 = self.LN_norm(img_sa1) + img_sa1_res*self.alpha
                img_sa2 = self.LN_norm(img_sa2) + img_sa2_res*self.alpha
                img_sa3 = self.LN_norm(img_sa3) + img_sa3_res*self.alpha

            # # bitemporal fusion
            if i == self.n_layers-1:
                img1_cap = img_sa1.transpose(-1, 1).view(batch, c, h, w)
                img2_cap = img_sa2.transpose(-1, 1).view(batch, c, h, w)
                img3_cap = img_sa3.transpose(-1, 1).view(batch, c, h, w)

                # feat_cap = torch.cat([img1_cap, img2_cap], dim=1)
                feat_cap = torch.cat([img1_cap, img2_cap, img3_cap], dim=1)
                # print(torch.cat([img1_cap, img2_cap], dim=1).shape)
                # print(feat_cap.shape)
                feat_cap = self.LN_list[i](self.Conv1_list[i](feat_cap))
                # feat_cap = self.Conv1_list[i](feat_cap)
                img_fuse = feat_cap.view(batch, c, -1).transpose(-1, 1)#.unsqueeze(-1) # (batch_size, L, D)
                img_fuse = self.LN_norm(img_fuse).unsqueeze(-1)
                img_list.append(img_fuse)

        # Out
        feat_cap = img_list[-1][..., 0]
        feat_cap = feat_cap.transpose(-1, 1)
        return feat_cap

if __name__ == '__main__':
    # test
    # img_A = torch.randn(16, 49, 768).cuda()
    # img_B = torch.randn(16, 49, 768).cuda()
    # img_C = torch.randn(16, 49, 768).cuda()
    # text_indices = torch.randint(0, 240, (16, 240)).cuda()
    # encoder2 = Encoder('CLIP-ViT-B/32').cuda()
    encoder2 = Encoder('resnet50').cuda()
    # encoder2 = Encoder('resnet101').cuda()
    max_length = 270
    text_indices = torch.randint(0, max_length, (4, max_length)).cuda()
    print(text_indices.max().item())
    img_A, img_B, img_C = encoder2(torch.randn(4,3, 224, 224).cuda(),torch.randn(4,3, 224, 224).cuda(),torch.randn(4,3, 224, 224).cuda())
    print(img_A.shape)

    encoder = AttentiveEncoder(n_layers=3, feature_size=(7, 7, 768), heads=8, max_length=max_length, vocab_size = max_length).cuda()
    feat_cap = encoder(img_A, img_B, img_C, text_indices)
    print(feat_cap.shape)
    # print(feat_cap)



    print('Done')
