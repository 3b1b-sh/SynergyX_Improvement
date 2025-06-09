# File: models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from .head import FusionHead
from .model_utlis import Embeddings, Encoder, EncoderCA, EncoderCell, EncoderCellCA, EncoderD2C, EncoderSSA, EncoderCellSSA

class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size=[2,2,6]
        drop_rate=0.2
        kernel_size=[16,16,16]

        if in_channel == 3:
            in_channels=[3,8,16]
            out_channels=[8,16,32]         

        elif in_channel == 6:
            in_channels=[6,16,32]
            out_channels=[16,32,64]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)

    def forward(self, x):
        # x: [batch, genes, channels]
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)  # [batch, out_channel, some_length]
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed) # [batch, some_length, feat_dim]
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)

def drug_feat(subs, subs_mask, device, patch, length):
    """
    subs: [batch, max_d] 的索引张量
    subs_mask: [batch, max_d] 的 mask 张量
    """
    subs = subs.long().to(device)
    subs_mask = subs_mask.long().to(device)

    if patch > length:
        padding = torch.zeros(subs.size(0), patch - length).long().to(device)
        subs = torch.cat((subs, padding), dim=1)
        subs_mask = torch.cat((subs_mask, padding), dim=1)

    expanded_subs_mask = subs_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_d]
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0

    return subs, expanded_subs_mask.float()

# <<< BATCH GNN START >>>
class GNNEncoder(nn.Module):
    """
    Batched GNN encoder. 输入是拼接后的所有节点 x_all、拼接后的所有边索引 edge_index_all
    以及 batch_index 标记哪些节点属于哪个子图，输出是 [batch_size, output_dim]。
    这里改为三层 GCN + BatchNorm + ReLU + Dropout。
    """

    def __init__(self, node_feat_dim=11, hidden_dim=128, output_dim=2560):
        super(GNNEncoder, self).__init__()
        # 三层 GCNConv
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # projection hidden_dim -> output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)  # 增大 dropout 至 0.3

    def forward(self, x_all, edge_index_all, batch_index):
        """
        x_all: [sum_nodes, node_feat_dim]
        edge_index_all: [2, sum_edges]
        batch_index: [sum_nodes]
        返回: [batch_size, output_dim]
        """
        x = x_all.float()

        # 第一层
        x = self.conv1(x, edge_index_all)      # [sum_nodes, hidden_dim]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层
        x = self.conv2(x, edge_index_all)      # [sum_nodes, hidden_dim]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第三层
        x = self.conv3(x, edge_index_all)      # [sum_nodes, hidden_dim]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 全局均值池化
        x_graph = global_mean_pool(x, batch_index)  # [batch_size, hidden_dim]

        # projection 到 output_dim
        out = self.fc(x_graph)  # [batch_size, output_dim]
        out = F.relu(out)
        out = self.dropout(out)
        return out  # [batch_size, output_dim]

# <<< BATCH GNN END >>>

class SynergyxNet(torch.nn.Module):

    def __init__(self,
                 num_attention_heads=8,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 max_length=50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 args=None):
        super(SynergyxNet, self).__init__()

        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.max_length = max_length

        if args.celldataset == 0:
            self.genes_nums = 697
        elif args.celldataset == 1:
            self.genes_nums = 18498
        elif args.celldataset == 2:
            self.genes_nums = 4079

        if self.args.cellencoder == 'cellTrans':
            self.patch = 50
            if self.in_channel == 3:
                feat_dim = 243
                hidden_size = 256
            elif self.in_channel == 6:
                feat_dim = 243*2
                hidden_size = 512
            self.cell_linear = nn.Linear(feat_dim, hidden_size)

        elif self.args.cellencoder == 'cellCNNTrans':
            self.patch = 165
            if self.in_channel == 3:
                hidden_size = 64
            elif self.in_channel == 6:
                hidden_size = 128 
            self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size, args=args)
        else:
            raise ValueError('Wrong cellencoder type!!!')

        intermediate_size = hidden_size * 2

        # ---------- 原有 1D Embedding ----------

        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)
        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_SA = EncoderCell(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_CA = EncoderCA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_CA = EncoderCellCA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_cell_CA = EncoderD2C(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.drug_SSA = EncoderSSA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.cell_SSA = EncoderCellSSA(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)

        self.head = FusionHead()

        self.cell_fc = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.drug_fc = nn.Sequential(
            nn.Linear(self.patch * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # ---------- ADD 2D GNN 初始化（批量版本）----------
        if args.use_2d == 1:
            # node_feat_dim = 11 (与 generate_drug_graph 保持一致), hidden_dim = 128, output_dim = 2560
            self.drugA_gnn = GNNEncoder(node_feat_dim=11, hidden_dim=128, output_dim=output_dim)
            self.drugB_gnn = GNNEncoder(node_feat_dim=11, hidden_dim=128, output_dim=output_dim)
        else:
            self.drugA_gnn = None
            self.drugB_gnn = None

        # 若同时启用 1D 与 2D，需要一个全连接把 2*output_dim 映射回 output_dim
        if args.use_1d == 1 and args.use_2d == 1 and args.use_fp == 0:
            self.fuse_fc = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        elif args.use_1d == 1 and args.use_2d == 0 and args.use_fp == 1:
            # 融合 1D + FP: 2*output_dim -> output_dim
            self.fuse_fc = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        elif args.use_1d == 0 and args.use_2d == 1 and args.use_fp == 1:
            # 融合 2D + FP: 2*output_dim -> output_dim
            self.fuse_fc = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        elif args.use_1d == 1 and args.use_2d == 1 and args.use_fp == 1:
            # 融合 1D + 2D + FP: 3*output_dim -> output_dim
            self.fuse_fc = nn.Sequential(
                nn.Linear(output_dim * 3, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        else:
            self.fuse_fc = None
        # ---------- ADD 2D GNN END ----------

        # ---------- ADD 2D Fingerprint 初始化 ----------
        if args.use_fp == 1:
            # 假设指纹长度是 args.fp_bits
            n_bits = getattr(args, 'fp_bits', 1024)
            # 一个简单的两层 MLP，把 [n_bits] 映射到 [output_dim]
            self.drugA_fp_fc = nn.Sequential(
                nn.Linear(n_bits, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
            self.drugB_fp_fc = nn.Sequential(
                nn.Linear(n_bits, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        else:
            self.drugA_fp_fc = None
            self.drugB_fp_fc = None
        # ---------- ADD 2D Fingerprint END ----------

    def forward(self, data):
        # 至少启用一项编码
        if self.args.use_1d == 0 and self.args.use_2d == 0 and self.args.use_fp == 0:
            raise RuntimeError("All of use_1d, use_2d and use_fp are set to 0. At least one must be 1.")

        # 确定 batch_size
        if self.args.mode == 'infer':
            batch_size = 1
        else:
            batch_size = self.args.batch_size

        # ---------- STEP 1：1D 编码 (若启用) ----------
        if self.args.use_1d == 1:
            # data.drugA: [batch, 1, max_d], data.drugA_mask: [batch, 1, max_d]
            drugA_idx = data.drugA.squeeze(1)       # [batch, max_d]
            drugA_mask = data.drugA_mask.squeeze(1) # [batch, max_d]
            drugB_idx = data.drugB.squeeze(1)
            drugB_mask = data.drugB_mask.squeeze(1)

            # 调用修改后的 drug_feat
            drugA, drugA_attention_mask = drug_feat(drugA_idx, drugA_mask, self.args.device, self.patch, self.max_length)
            drugB, drugB_attention_mask = drug_feat(drugB_idx, drugB_mask, self.args.device, self.patch, self.max_length)

            drugA_1d = self.drug_emb(drugA)
            drugB_1d = self.drug_emb(drugB)
            drugA_1d = drugA_1d.float()
            drugB_1d = drugB_1d.float()

            x_cell = data.x_cell.type(torch.float32)
            x_cell = x_cell[:, [self.omic_dict[i] for i in self.include_omic]]  # [batch*genes_nums, len(omics)]
            cellA = x_cell.view(batch_size, self.genes_nums, -1)
            cellB = cellA.clone()

            if self.args.cellencoder == 'cellTrans':
                gene_length = 4050
                cellA = cellA[:, :gene_length, :]
                cellA = cellA.view(batch_size, self.patch, -1, x_cell.size(-1))
                cellA = cellA.view(batch_size, self.patch, -1)
                cellA = self.cell_linear(cellA)
                cellB = cellB[:, :gene_length, :]
                cellB = cellB.view(batch_size, self.patch, -1, x_cell.size(-1))
                cellB = cellB.view(batch_size, self.patch, -1)
                cellB = self.cell_linear(cellB)
            elif self.args.cellencoder == 'cellCNNTrans':
                cellA = self.cell_conv(cellA)
                cellB = self.cell_conv(cellB)

            cellA0 = cellA
            cellA, drugA_1d, attn9, attn10 = self.drug_cell_CA(cellA, drugA_1d, drugA_attention_mask, None)
            cellB, drugB_1d, attn11, attn12 = self.drug_cell_CA(cellB, drugB_1d, drugB_attention_mask, None)
            cellA1 = cellA
            cellB1 = cellB

            drugA_1d, attn5 = self.drug_SA(drugA_1d, drugA_attention_mask)
            drugB_1d, attn6 = self.drug_SA(drugB_1d, drugB_attention_mask)
            cellA, attn7 = self.cell_SA(cellA, None)
            cellB, attn8 = self.cell_SA(cellB, None)
            cellA2 = cellA
            cellB2 = cellB

            drugA_1d, drugB_1d, attn1, attn2 = self.drug_CA(drugA_1d, drugB_1d, drugA_attention_mask, drugB_attention_mask)
            cellA, cellB, attn3, attn4 = self.cell_CA(cellA, cellB, None, None)
            cellA3 = cellA
            cellB3 = cellB

            # 1D embedding -> [batch_size, output_dim]
            drugA_embed_1d = self.drug_fc(drugA_1d.view(-1, drugA_1d.shape[1] * drugA_1d.shape[2]))
            drugB_embed_1d = self.drug_fc(drugB_1d.view(-1, drugB_1d.shape[1] * drugB_1d.shape[2]))

            # 细胞系 embedding
            cellA_embed = self.cell_fc(cellA.view(-1, cellA.shape[1] * cellA.shape[2]))
            cellB_embed = self.cell_fc(cellB.view(-1, cellB.shape[1] * cellB.shape[2]))
            cell_embed = torch.cat((cellA_embed, cellB_embed), 1)  # [batch_size, output_dim*2]
        else:
            # 若未启用 1D，只构造 cell_embed
            drugA_embed_1d = None
            drugB_embed_1d = None
            x_cell = data.x_cell.type(torch.float32)
            x_cell = x_cell[:, [self.omic_dict[i] for i in self.include_omic]]
            cellA = x_cell.view(batch_size, self.genes_nums, -1)
            cellB = cellA.clone()
            if self.args.cellencoder == 'cellTrans':
                gene_length = 4050
                cellA = cellA[:, :gene_length, :]
                cellA = cellA.view(batch_size, self.patch, -1, x_cell.size(-1))
                cellA = cellA.view(batch_size, self.patch, -1)
                cellA = self.cell_linear(cellA)
                cellB = cellB[:, :gene_length, :]
                cellB = cellB.view(batch_size, self.patch, -1, x_cell.size(-1))
                cellB = cellB.view(batch_size, self.patch, -1)
                cellB = self.cell_linear(cellB)
            elif self.args.cellencoder == 'cellCNNTrans':
                cellA = self.cell_conv(cellA)
                cellB = self.cell_conv(cellB)
            cellA_embed = self.cell_fc(cellA.view(-1, cellA.shape[1] * cellA.shape[2]))
            cellB_embed = self.cell_fc(cellB.view(-1, cellB.shape[1] * cellB.shape[2]))
            cell_embed = torch.cat((cellA_embed, cellB_embed), 1)  # [batch_size, output_dim*2]
            cellA0 = cellA
            cellA1 = cellA
            cellB1 = cellB
            cellA2 = cellA
            cellB2 = cellB
            cellA3 = cellA
            cellB3 = cellB
            attn1 = attn2 = attn3 = attn4 = attn5 = attn6 = attn7 = attn8 = attn9 = attn10 = attn11 = attn12 = None

        # ---------- STEP 2：2D GNN 批量编码 (若启用) ----------
        if self.args.use_2d == 1:
            # 取出拼接后的所有节点特征和边索引
            drugA_x_all = data.drugA_x.to(device=self.args.device, dtype=torch.float32)            # [sum_nodes_A, feat_dim=10]
            drugA_edge_index_all = data.drugA_edge_index.to(device=self.args.device, dtype=torch.long)  # [2, sum_edges_A]
            num_nodes_list_A = data.drugA_num_nodes       # e.g., Tensor([n0, n1, ..., n_{batch_size-1}])
            num_edges_list_A = data.drugA_num_edges       # e.g., Tensor([e0, e1, ..., e_{batch_size-1}])

            # 确保它们是 Python 列表
            if isinstance(num_nodes_list_A, torch.Tensor):
                num_nodes_list_A = num_nodes_list_A.tolist()
            if isinstance(num_edges_list_A, torch.Tensor):
                num_edges_list_A = num_edges_list_A.tolist()

            # 构造每个子图的 PyG Data 对象列表
            drugA_graph_list = []
            node_ptr = 0
            edge_ptr = 0
            for i in range(batch_size):
                n_i = num_nodes_list_A[i]
                e_i = num_edges_list_A[i]
                # 提取第 i 个子图的节点特征
                x_i = drugA_x_all[node_ptr: node_ptr + n_i]  # [n_i, feat_dim=10]
                # 提取并偏移第 i 个子图的边索引
                edge_i = drugA_edge_index_all[:, edge_ptr: edge_ptr + e_i]  # [2, e_i]
                # 构建 PyG Data
                data_i = Data(x=x_i, edge_index=edge_i)
                drugA_graph_list.append(data_i)
                node_ptr += n_i
                edge_ptr += e_i

            # 用 PyG Batch 把所有子图打包
            batchA = Batch.from_data_list(drugA_graph_list)
            # 调用 GNNEncoder，得到 [batch_size, output_dim]
            drugA_embed_2d = self.drugA_gnn(batchA.x.to(self.args.device),
                                             batchA.edge_index.to(self.args.device),
                                             batchA.batch.to(self.args.device))

            # 同理处理 drugB
            drugB_x_all = data.drugB_x.to(device=self.args.device, dtype=torch.float32)
            drugB_edge_index_all = data.drugB_edge_index.to(device=self.args.device, dtype=torch.long)
            num_nodes_list_B = data.drugB_num_nodes
            num_edges_list_B = data.drugB_num_edges
            if isinstance(num_nodes_list_B, torch.Tensor):
                num_nodes_list_B = num_nodes_list_B.tolist()
            if isinstance(num_edges_list_B, torch.Tensor):
                num_edges_list_B = num_edges_list_B.tolist()

            drugB_graph_list = []
            node_ptr = 0
            edge_ptr = 0
            for i in range(batch_size):
                n_i = num_nodes_list_B[i]
                e_i = num_edges_list_B[i]
                x_i = drugB_x_all[node_ptr: node_ptr + n_i]
                edge_i = drugB_edge_index_all[:, edge_ptr: edge_ptr + e_i]
                data_i = Data(x=x_i, edge_index=edge_i)
                drugB_graph_list.append(data_i)
                node_ptr += n_i
                edge_ptr += e_i

            batchB = Batch.from_data_list(drugB_graph_list)
            drugB_embed_2d = self.drugB_gnn(batchB.x.to(self.args.device),
                                             batchB.edge_index.to(self.args.device),
                                             batchB.batch.to(self.args.device))
        else:
            # 如果没启用 2D，用全零张量占位
            device = self.args.device
            drugA_embed_2d = torch.zeros((batch_size, 2560), device=device)
            drugB_embed_2d = torch.zeros_like(drugA_embed_2d)

        # ---------- STEP 2.5：2D Fingerprint 编码 (若启用) ----------
        if self.args.use_fp == 1:
            # data.drugA_fp: [batch, n_bits], data.drugB_fp: [batch, n_bits]
            drugA_fp_raw = data.drugA_fp.to(self.args.device, dtype=torch.float32)  # [batch, n_bits]
            drugB_fp_raw = data.drugB_fp.to(self.args.device, dtype=torch.float32)
            drugA_embed_fp = self.drugA_fp_fc(drugA_fp_raw)  # [batch, output_dim]
            drugB_embed_fp = self.drugB_fp_fc(drugB_fp_raw)
        else:
            # 未启用时占位全零
            device = self.args.device
            drugA_embed_fp = torch.zeros((batch_size, 2560), device=device)
            drugB_embed_fp = torch.zeros_like(drugA_embed_fp)

        # ---------- STEP 3：融合或选用 1D/2D/FP Embedding ----------
        embeddings_to_concat_A = []
        embeddings_to_concat_B = []

        if self.args.use_1d == 1:
            embeddings_to_concat_A.append(drugA_embed_1d)
            embeddings_to_concat_B.append(drugB_embed_1d)
        if self.args.use_2d == 1:
            embeddings_to_concat_A.append(drugA_embed_2d)
            embeddings_to_concat_B.append(drugB_embed_2d)
        if self.args.use_fp == 1:
            embeddings_to_concat_A.append(drugA_embed_fp)
            embeddings_to_concat_B.append(drugB_embed_fp)

        # 将选中的分支沿 dim=1 拼接
        drugA_cat = torch.cat(embeddings_to_concat_A, dim=1)  # [batch, k*output_dim]
        drugB_cat = torch.cat(embeddings_to_concat_B, dim=1)  # [batch, k*output_dim]

        # 若启用了多于一个分支，则做 fuse_fc 将维度降回 output_dim；否则直接使用单分支输出
        if len(embeddings_to_concat_A) > 1:
            drugA_embed = self.fuse_fc(drugA_cat)  # [batch, output_dim]
            drugB_embed = self.fuse_fc(drugB_cat)
        else:
            # 只有一个分支时直接取拼接结果（即本身已是 output_dim）
            drugA_embed = drugA_cat
            drugB_embed = drugB_cat

        # 拼接 A/B，得到 [batch_size, 2*output_dim]
        drug_embed = torch.cat((drugA_embed, drugB_embed), dim=1)

        # ---------- STEP 4：送入 FusionHead 得到最终预测输出 ----------
        output = self.head(cell_embed, drug_embed)

        all_cell_embed = None
        all_attn = None
        if self.args.output_attn:
            all_cell_embed = torch.stack((
                cellA1.mean(axis=1).flatten(),
                cellB1.mean(axis=1).flatten(),
                cellA2.mean(axis=1).flatten(),
                cellB2.mean(axis=1).flatten(),
                cellA3.mean(axis=1).flatten(),
                cellB3.mean(axis=1).flatten(),
                cellA0.mean(axis=1).flatten(),
            ), 0)
            all_attn = torch.cat((attn9, attn10, attn11, attn12,
                                  attn5, attn6, attn7, attn8,
                                  attn1, attn2, attn3, attn4), 0)

        return output, all_cell_embed, all_attn

    def init_weights(self):
        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
        if self.args.cellencoder == 'cellCNNTrans':
            self.cell_conv.init_weights()
