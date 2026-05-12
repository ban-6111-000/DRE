# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from sympy import true
from .basic_layers import Transformer, GradientReversalLayer
from einops import rearrange, repeat
from models.transformers_encoder.transformer import TransformerEncoder
from scipy.spatial.distance import squareform
from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from torch_geometric.nn import GATConv
from sklearn.manifold import TSNE
import seaborn as sns

def edge_perms(l, window_past, window_future):

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)

def batch_graphify(features, qmask, lengths, window_past, window_future, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification 
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method 
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from 
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces 
    from any other dialogue instances in that mini-batch.
    """
    edge_index, edge_type, node_features = [], [], []
    edge_index_modal = []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []   

    for j in range(batch_size):
        node_features.append(features[j,:lengths[j].item(), :])
        perms1 = edge_perms(lengths[j].item(), window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j].item()
        edge_index_lengths.append(len(perms1))
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)

    edge_index_ =  torch.stack([edge_index[0] + node_features.shape[0], edge_index[1] + node_features.shape[0]],dim=0)
    for i in range(node_features.shape[0]):
        edge_index_modal.append(torch.tensor([i, i+node_features.shape[0]]))
        edge_index_modal.append(torch.tensor([i+node_features.shape[0], i]))
    edge_index_modal_ = torch.stack(edge_index_modal).transpose(0, 1)

    edge_index1 = torch.cat([edge_index,edge_index_,edge_index_modal_], dim=-1)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_index1 = edge_index1.cuda()

    return node_features, edge_index, edge_index_lengths, edge_index1

def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(0)

    for j in range(batch_size):
        node_features.append(features[j,:lengths[j].item(), :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features
#新

class SimGCL(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, emb):
        norm = torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-9
        noise = torch.sign(emb) / norm
        return emb + self.eps * noise

class JO(GeneralRecommender): #数据相同，线上模型过MLP,目标模型丢弃
    def __init__(self, config, dataset):
        super(JO, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.jo_weight = config['jo_weight']
        self.dropout = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(
            dataset.inter_matrix(form='coo').astype(np.float32)
        ).coalesce().to(self.device)

        self.noise_eps = 0.2  # SimGCL默认 0.1~0.3 之间最稳   # 可改
        #得eu,ei
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        #初始化相应嵌入层的权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

# 预测层
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        # L2正则化
        self.reg_loss = EmbLoss()
        #初始化predictor网络的权重
        nn.init.xavier_normal_(self.predictor.weight)

       #7050,4096/384
        #改图像，文本输入维度  统一64维
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False) # (batchsize,4096)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)# (batchsize,64)
            nn.init.xavier_normal_(self.image_trs.weight)
            #self.image_trs2 = nn.Linear(self.t_feat.shape[1], 50)  # 384 -> 50
            print("----------")
            print(self.image_embedding)
            print(self.image_trs)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)# (batchsize,384)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)# (batchsize,64)
            nn.init.xavier_normal_(self.text_trs.weight)
            #self.text_trs2 = nn.Linear(self.v_feat.shape[1], 500)  # 4096 -> 500

        #==============================
        token = config['token_length']  # 8
        feat_embed_dim = config['embed_dim']  # 128
        
# (1, 8, 64)
        self.h_pt = nn.Parameter(torch.ones(1, token, feat_embed_dim))
        self.h_pv = nn.Parameter(torch.ones(1, token, feat_embed_dim))

# 文本投影
        self.proj_t = nn.Sequential(
            nn.Linear(384, 128),
            Transformer(
            num_frames=50,
            save_hidden=False,
            token_len=token,
            dim=feat_embed_dim,
            depth=2,
            heads=config['heads'],
            mlp_dim=feat_embed_dim
        ))

# 图像投影
        self.proj_v = nn.Sequential(
        nn.Linear(4096, 1024),       # 阶段降维
        Transformer(
            num_frames=10,
            save_hidden=False,
            token_len=token,
            dim=1024,
            depth=2,
            heads=config['heads'],
            mlp_dim=1024
        ),
         nn.Linear(1024, 128)
    )

        self.token = token
        self.feat_embed_dim = feat_embed_dim

# 做最终预测用 64->1   
        self.dmml = nn.ModuleList([
            nn.Linear(config['input_dim'], config['output_dim'])
        ])
# 非共性解码器  (batchsize,8,64)
        self.encoder_s_t = self.get_network(self_type='t', layers = 2)       
        self.encoder_s_v = self.get_network(self_type='v', layers = 2)
# 共性解码器
        self.encoder_c = self.get_network(self_type='t', layers = 2)   

        self.mse_loss = nn.MSELoss(reduction='mean')
# 图融合        
        '''self.graph = GAT(nfeat=384,
                         nhid=8,
                         nclass=128,  # 最后的类别 相当于下一层的输入
                         dropout=0.6,
                         nheads=3,  # 之前是8个
                         alpha=0.2
                         )
        '''
        improvedgatlayer_vt = ImprovedGATLayer(200, dropout=0, num_heads=4, use_residual=True, no_cuda=False)
        self.improvedgat_vt = ImprovedGAT(improvedgatlayer_vt, num_layers=5, hidesize=200)
        self.fusion_vt = ConcatFusion(len('vt'), input_dim=2*200, output_dim=128)
        
        # 为了把batchsize变为64
        #改
        self.fc_out = nn.Linear(200, 128)


    def get_network(self, self_type='t', layers=-1):
        if self_type in ['t', 'vt']:
            embed_dim, attn_dropout = 128, 0.3
        elif self_type in ['v', 'tv']:
            embed_dim, attn_dropout = 128, 0.0
         
        elif self_type == 't_mem':
            embed_dim, attn_dropout = 128, 0.3
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 128, 0.3
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=8,
                                  layers=max(2, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=0.0,
                                  res_dropout=0.0,
                                  embed_dropout=0.2,
                                  attn_mask=true)
    
    def sim_operation_between_2(self, x0, x1):
        return self.calc_sim(x0, x1)  
    
    def calc_sim(self, x1, x2):
        return self.mse_loss(x1,x2)
    
    def calc_diff_loss(self, diff_list):
        for i in range(len(diff_list)):
            for j in range(i+1,len(diff_list)):
                if i == 0 and j == 1:
                    loss = torch.mean(torch.abs(torch.cosine_similarity(diff_list[i], diff_list[j], dim=-1)))
                else:
                    loss = loss + torch.mean(torch.abs(torch.cosine_similarity(diff_list[i], diff_list[j], dim=-1)))
        return loss

    def get_norm_adj_mat(self, interaction_matrix): #得 L = D * A * D
        # A 稀疏矩阵
        A = sp.dok_matrix((self.n_users + self.n_items,
                        self.n_users + self.n_items), dtype=np.float32)
    
        # M 为交互阵 M_t 转置
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
    
        # 创建行列对，数据字典的每个键是 (row, col)，每个值是 1
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                         [1] * inter_M.nnz))
    
        # 将转置矩阵中的行列对也加入到 data_dict 中
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                              [1] * inter_M_t.nnz)))
    
        # 直接通过索引方式更新 A，而不是使用 _update()
        for (row, col), value in data_dict.items():
            A[row, col] = value
    
        # 计算标准化的邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        # 为了避免除以零的警告，给对角线加上 epsilon
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        # ******
        L = D * A * D
     
        # 将标准化的邻接矩阵转为稀疏矩阵 (增效率)
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self): # 得最终的hu,hi  
        h_item = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers): # 图卷积
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        # ===== 视图1 =====
        norm_adj_1 = self.norm_adj  # 无噪声
        ego_1 = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_1 = [ego_1]
        for i in range(self.n_layers):
            ego_1 = torch.sparse.mm(norm_adj_1, ego_1)
            all_1.append(ego_1)
        all_1 = torch.stack(all_1, dim=1).mean(dim=1)
        u1, i1 = torch.split(all_1, [self.n_users, self.n_items], dim=0)
        i1 = i1 + h_item  # 残差

        # ===== 视图2（SimGCL增强视图）=====
        norm_adj_2 = self.add_graph_noise(self.norm_adj)  # ★加噪
        ego_2 = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_2 = [ego_2]
        for i in range(self.n_layers):
            ego_2 = torch.sparse.mm(norm_adj_2, ego_2)
            all_2.append(ego_2)
        all_2 = torch.stack(all_2, dim=1).mean(dim=1)
        u2, i2 = torch.split(all_2, [self.n_users, self.n_items], dim=0)
        i2 = i2 + h_item  # 残差
#===============
#改
        batch_size = 128

        txt_vec_seq = self.t_feat[:batch_size].unsqueeze(1).repeat(1, 50, 1)  # [64, 50, 384]
        img_vec_seq = self.v_feat[:batch_size].unsqueeze(1).repeat(1, 10, 1) # [64, 10, 4096]



# 文本,图像投影
        h_1_t = self.proj_t(txt_vec_seq)[:, :8]  # [batch, 50, hidden_dim]->[batch, 8, 128]
        h_1_v = self.proj_v(img_vec_seq)[:, :8]  # [batch, 10, hidden_dim]->[batch, 8, 128]

        # 共性编码器
        c_t = self.encoder_c(h_1_t)
        c_v = self.encoder_c(h_1_v)
        # 非共性编码器
        s_t = self.encoder_s_t(h_1_t)    
        s_v = self.encoder_s_v(h_1_v)

        batch_size,seq_len,_=c_t.shape
        umask = torch.ones((batch_size, seq_len), dtype=torch.float32).to(self.device)
        lengths0 = []
        for j, umask_ in enumerate(umask):
            lengths0.append((umask[j] == 1).nonzero()[-1][0] + 1)
        seq_lengths = torch.stack(lengths0)

        features_c_t, edge_index_c_t, _, edge_index1_c_l = batch_graphify(c_t, None, seq_lengths, 16, 16, False)
        features_c_v = simple_batch_graphify(c_v, seq_lengths, False)
        features_s_t = simple_batch_graphify(s_t, seq_lengths, False)
        features_s_v = simple_batch_graphify(s_v, seq_lengths, False)

        features_single_cvt = torch.cat([features_c_v, features_c_t], dim=0)
        features_cross_cvt = self.improvedgat_vt(features_single_cvt, edge_index1_c_l)
        features_cross_cv2, features_cross_ct2 = torch.chunk(features_cross_cvt, 2, dim=0)

        features_single_svt = torch.cat([features_s_v, features_s_t], dim=0)
        features_cross_svt = self.improvedgat_vt(features_single_svt, edge_index1_c_l)
        features_cross_sv2, features_cross_st2 = torch.chunk(features_cross_svt, 2, dim=0)

        shared_t =  features_cross_ct2  # 1024 200
        shared_v =  features_cross_cv2
        
        specil_t =  features_cross_st2
        specil_v =  features_cross_sv2


        shared_t = self.fc_out(shared_t)  # 1024 128
        shared_t = shared_t.view(batch_size, 8, 128)  #128,8,128
        shared_v = self.fc_out(shared_v)
        shared_v = shared_v.view(batch_size, 8, 128)

        specil_t = self.fc_out(specil_t)
        specil_t = specil_t.view(batch_size, 8, 128)
        specil_v = self.fc_out(specil_v)
        specil_v = specil_v.view(batch_size, 8, 128)

        recon_t = shared_t + specil_t
        recon_v = shared_v + specil_v

        c_t_r = self.encoder_c(recon_t)
        c_v_r = self.encoder_c(recon_v)

        s_t_r = self.encoder_s_t(recon_t)    
        s_v_r = self.encoder_s_v(recon_v)


        share_sim_loss = self.sim_operation_between_2(c_t,c_v)
        sup_sim_loss = (self.calc_sim(c_t, c_t_r) + \
                                self.calc_sim(c_v, c_v_r)+ \
                                self.calc_sim(s_t, s_t_r) + \
                                self.calc_sim(s_v, s_v_r)) 
        diff_list = [c_t,c_v,s_t,s_v]
        diff_loss = self.calc_diff_loss(diff_list)
#specil_t,specil_v解耦模态
        return (
    u_g_embeddings, i_g_embeddings, (u1, i1), (u2, i2),
    {
        'share_sim_loss': share_sim_loss,
        'sup_sim_loss': sup_sim_loss,
        'diff_loss': diff_loss
    },{'s_t':specil_t,
       's_v':specil_v,
       's_h':shared_t},
)

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori, (u1, i1), (u2, i2), loss_dict,features= self.forward()
        share_sim_loss = loss_dict['share_sim_loss']
        sup_sim_loss = loss_dict['sup_sim_loss']
        diff_loss = loss_dict['diff_loss']
        jo_loss = share_sim_loss + sup_sim_loss + diff_loss
        t_feat_online, v_feat_online = None, None
        # 线性嵌入
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad(): # 停止梯度
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            # 丢弃
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        # 行为user，列为item 提出来存到对应数组中
        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None: # 算v模态的
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            # 用负余弦相似度得 Lalign
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            # 用负余弦相似度得 Lmask
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None: # 算t模态的
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        #算Lrec
        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        users, items = interactions[0], interactions[1]

        u1_batch = u1[users]
        u2_batch = u2[users]
        i1_batch = i1[items]
        i2_batch = i2[items]

        # ==== 2. 用户与物品的SimGCL对比损失 ====
        def info_nce(z1, z2, temp=0.2):
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            pos = (z1 * z2).sum(dim=1) / temp
            sim = torch.matmul(z1, z2.t()) / temp

            # 屏蔽对角线（不能当负样本）
            mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
            sim = sim.masked_fill(mask, -1e9)

            neg = torch.logsumexp(sim, dim=1)
            return -(pos - neg).mean()


        # ★ 用户对比
        loss_user = info_nce(u1_batch, u2_batch)
        # ★ 商品对比
        loss_item = info_nce(i1_batch, i2_batch)

        #算Lrec
        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        # reg
        reg = self.reg_weight * self.reg_loss(u1, i1)

        return (loss_ui + loss_iu ).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
                self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean() + self.jo_weight * jo_loss + 0.001 * (loss_user + loss_item) + reg
        #

    def full_sort_predict(self, interaction): # 得分数
        user = interaction[0]
        u_online, i_online, _, _, _,_,_,_ = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui
    
    def add_graph_noise(self, adj):
        # adj: torch.sparse.FloatTensor
        noise = torch.randn_like(adj._values()) * self.noise_eps
        new_values = adj._values() + noise
        return torch.sparse.FloatTensor(adj._indices(), new_values, adj.size())
    

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)    # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)   # elu-激活函数
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
        
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                           concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, x, adj):
        #x = F.dropout(x, self.dropout, training=self.training)
        print(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # 第二层的attention layer
        return F.log_softmax(x, dim=1)
        
class ImprovedGATLayer(torch.nn.Module):
    def __init__(self, hidesize, dropout=0.5, num_heads=5, use_residual=True, no_cuda=False):
        super(ImprovedGATLayer, self).__init__()
        self.no_cuda = no_cuda
        self.use_residual = use_residual
        self.convs = GATConv(hidesize, hidesize, heads=num_heads, add_self_loops=True, concat=False)

    def forward(self, features, edge_index):
        x = features
        if self.use_residual:
            x = x + self.convs(x, edge_index)
        else:
            x = self.convs(x, edge_index)

        return x

class ImprovedGAT(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers, hidesize):
        super(ImprovedGAT, self).__init__()
        layer = []
        for l in range(num_layers):
            layer.append(encoder_layer)
        self.layers = nn.ModuleList(layer)
        self.out_mlp = nn.Linear((num_layers+1)*hidesize, hidesize)
        #**
        self.input_proj = nn.Linear(128, 200)

    def forward(self, features, edge_index):
        features = self.input_proj(features)
        out = features
        output = [out]
        for mod in self.layers:
            out = mod(out, edge_index)
            output.append(out)
        output_ = torch.cat(output, dim=-1)
        output_ = self.out_mlp(output_)
        return output_
class ConcatFusion(nn.Module):
    def __init__(self, len_modals, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.len_modals = len_modals
    def forward(self, x, y, z):
        if self.len_modals ==2:
            output = torch.cat((x, y), dim=1)
            output = self.fc_out(output)
            return output
        if self.len_modals ==3:
            output = torch.cat((x, y, z), dim=1)
            output = self.fc_out(output)
            return output
