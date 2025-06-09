# File: dataset/My_inMemory_dataset.py

import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .base_InMemory_dataset import BaseInMemoryDataset

class MyInMemoryDataset(BaseInMemoryDataset):
    """
    基于 1D 子结构、2D GNN、2D 指纹 三种可选编码，生成 PyG Dataset。
    args 中使用：
      - args.use_1d: 0/1, 是否启用 1D 子结构嵌入
      - args.use_2d: 0/1, 是否启用 2D GNN 图嵌入
      - args.use_fp: 0/1, 是否启用 2D 指纹编码
      - args.fp_bits: 指纹维度，如 1024
      其它参数和原有逻辑保持一致。
    """

    def __init__(self,
                 data_root,
                 data_items,
                 celllines_data,
                 drugs_data,
                 dgi_data=None,
                 transform=None,
                 pre_transform=None,
                 args=None,
                 max_node_num=155):
        super(MyInMemoryDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)

        # 构造 name，用于 processed 文件名
        if args.celldataset == 1:
            base_name = osp.basename(data_items).split('items')[0] + '18498g'
        elif args.celldataset == 2:
            base_name = osp.basename(data_items).split('items')[0] + '4079g'
        elif args.celldataset == 3:
            base_name = osp.basename(data_items).split('items')[0] + '963g'
        else:
            base_name = osp.basename(data_items).split('items')[0] + 'unknown'

        # 拼接各编码方式到 name
        encs = []
        if args.use_1d == 1:
            encs.append("1D")
        if args.use_2d == 1:
            encs.append("2DGN")
        if args.use_fp == 1:
            encs.append("2DFP")
        self.name = base_name + "_" + "_".join(encs) + "_norm"

        if args.mode == 'infer':
            # Infer 模式可以不加后缀
            self.name = osp.basename(data_items).split('items')[0]

        self.args = args

        # 加载 data_items 列表
        self.data_items = np.load(data_items, allow_pickle=True)
        # 加载细胞特征字典
        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        # 加载 1D 子结构嵌入字典： drug_smiles -> (indices_array, mask_array)
        self.drugs = np.load(drugs_data, allow_pickle=True).item()

        # 可选的 DGI 数据
        if dgi_data:
            self.dgi = np.load(dgi_data, allow_pickle=True).item()
        else:
            self.dgi = {}

        # ---------- 2D GNN 部分：加载图字典 ----------
        graph_path = osp.join(data_root, '1_drug_data', 'drug_graph_dict.npy')
        if not osp.exists(graph_path):
            raise FileNotFoundError(f"Drug graph dict not found at {graph_path}. "
                                    "请先运行 generate_drug_graph.py 生成文件。")
        self.drug_graph_dict = np.load(graph_path, allow_pickle=True).item()

        # ---------- 2D Fingerprint 部分：根据 use_fp 决定是否加载 ----------
        if args.use_fp == 1:
            fp_path = osp.join(data_root, f'1_drug_data/drug_fp_dict_{args.fp_bits}bits.npy')
            if not osp.exists(fp_path):
                raise FileNotFoundError(f"Drug fingerprint dict not found at {fp_path}. "
                                        "请先运行 generate_drug_fingerprint.py 生成文件。")
            self.drug_fp_dict = np.load(fp_path, allow_pickle=True).item()
        else:
            self.drug_fp_dict = {}

        self.max_node_num = max_node_num

        # 如果已经存在预处理后的文件，则直接加载
        if os.path.isfile(self.processed_paths[0]):
            print(f'Pre-processed data found: {self.processed_paths[0]}, loading ...')
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(f'Pre-processed data {self.processed_paths[0]} not found, doing pre-processing ...')
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # 返回一个固定的文件名： self.name + '.pt'
        return [self.name + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        data_list = []
        data_len = len(self.data_items)

        for i in tqdm(range(data_len)):
            drugA, drugB, c1, label = self.data_items[i]
            cell_features = self.celllines[c1]  # ndarray [num_genes, num_omics]

            # DGI (drug–gene interaction) 数据，没有则全 1
            dgiA = self.dgi.get(drugA, np.ones(cell_features.shape[0]))
            dgiB = self.dgi.get(drugB, np.ones(cell_features.shape[0]))

            # ========= 1D 子结构嵌入 =========
            if self.args.use_1d == 1:
                if drugA not in self.drugs or drugB not in self.drugs:
                    # 若缺失 1D 嵌入，则跳过该样本
                    continue
                drugA_features = self.drugs[drugA]  # (indices_array, mask_array)
                drugB_features = self.drugs[drugB]

            # ========= 2D GNN 图嵌入 =========
            if self.args.use_2d == 1:
                if drugA not in self.drug_graph_dict or drugB not in self.drug_graph_dict:
                    continue

                # 取出字典中的两个字段，而不是直接解包键
                entryA = self.drug_graph_dict[drugA]
                node_feat_A = entryA["node_feat"]       # numpy.ndarray [num_atoms_A, feat_dim]
                edge_index_A = entryA["edge_index"]     # numpy.ndarray [2, num_edges_A]

                entryB = self.drug_graph_dict[drugB]
                node_feat_B = entryB["node_feat"]       # numpy.ndarray [num_atoms_B, feat_dim]
                edge_index_B = entryB["edge_index"]     # numpy.ndarray [2, num_edges_B]

                # 从 node_feat 推算 num_nodes
                num_nodes_A = node_feat_A.shape[0]
                num_nodes_B = node_feat_B.shape[0]
                # edge_index 的第二维是 num_edges
                num_edges_A = edge_index_A.shape[1]
                num_edges_B = edge_index_B.shape[1]

                # 转为 torch.Tensor
                drugA_x = torch.tensor(node_feat_A, dtype=torch.float32)          # [num_nodes_A, feat_dim]
                drugA_edge_index = torch.tensor(edge_index_A, dtype=torch.long)   # [2, num_edges_A]

                drugB_x = torch.tensor(node_feat_B, dtype=torch.float32)          # [num_nodes_B, feat_dim]
                drugB_edge_index = torch.tensor(edge_index_B, dtype=torch.long)   # [2, num_edges_B]
            else:
                drugA_x = None
                drugA_edge_index = None
                num_nodes_A = None
                num_edges_A = None

                drugB_x = None
                drugB_edge_index = None
                num_nodes_B = None
                num_edges_B = None


            # ========= 2D Fingerprint =========
            if self.args.use_fp == 1:
                if drugA not in self.drug_fp_dict or drugB not in self.drug_fp_dict:
                    continue
                vecA_fp = self.drug_fp_dict[drugA]  # ndarray [n_bits,]
                vecB_fp = self.drug_fp_dict[drugB]
                # 转为 torch.Tensor，并加上 batch 维度
                drugA_fp = torch.tensor(vecA_fp, dtype=torch.float32).unsqueeze(0)  # [1, n_bits]
                drugB_fp = torch.tensor(vecB_fp, dtype=torch.float32).unsqueeze(0)  # [1, n_bits]
            else:
                drugA_fp = None
                drugB_fp = None

            # 构造 PyG Data 对象
            data_item = Data()

            # 1D 子结构 embedding
            if self.args.use_1d == 1:
                idxA, maskA = drugA_features
                idxB, maskB = drugB_features
                data_item.drugA = torch.tensor(np.array([idxA]), dtype=torch.long)      # [1, max_d]
                data_item.drugB = torch.tensor(np.array([idxB]), dtype=torch.long)
                data_item.drugA_mask = torch.tensor(np.array([maskA]), dtype=torch.long)  # [1, max_d]
                data_item.drugB_mask = torch.tensor(np.array([maskB]), dtype=torch.long)

            # 2D GNN 图
            if self.args.use_2d == 1:
                data_item.drugA_x = drugA_x            # [num_nodes_A, feat_dim]
                data_item.drugA_edge_index = drugA_edge_index  # [2, num_edges_A]
                data_item.drugA_num_nodes = num_nodes_A       # int
                data_item.drugA_num_edges = num_edges_A       # int

                data_item.drugB_x = drugB_x
                data_item.drugB_edge_index = drugB_edge_index
                data_item.drugB_num_nodes = num_nodes_B
                data_item.drugB_num_edges = num_edges_B

            # 2D Fingerprint
            if self.args.use_fp == 1:
                data_item.drugA_fp = drugA_fp  # [1, n_bits]
                data_item.drugB_fp = drugB_fp

            # 细胞系部分
            data_item.x_cell = torch.tensor(cell_features, dtype=torch.float32)  # [num_genes, num_omics]
            data_item.y = torch.tensor([float(label)], dtype=torch.float32)

            # DGI
            data_item.dgiA = torch.tensor(dgiA, dtype=torch.float32)
            data_item.dgiB = torch.tensor(dgiB, dtype=torch.float32)

            data_list.append(data_item)

        # 过滤 & transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Data pre-processing done. Saving to file ...')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Dataset construction done.')
