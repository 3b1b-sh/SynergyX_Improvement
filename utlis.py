import os
import os.path as osp
import random
import numpy as np
import torch
# from mmcv.utils import collect_env as collect_base_env
from torch_geometric.loader import DataLoader
from dataset.My_inMemory_dataset import MyInMemoryDataset
from metrics import get_metrics


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None):
        """
        Initialize EarlyStopping object.

        Args:
            mode (str): 'higher' if a higher score is better, 'lower' if a lower score is better.
            patience (int): Number of epochs to wait for improvement before early stopping.
            filename (str): Name of the checkpoint file to save the model state.
            metric (str): Metric to monitor for early stopping. Can be 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', or 'mse'.
            n_fold (int): Fold number used for naming checkpoint file.
            folder (str): Folder path to save checkpoint file.
        """

        if filename is None:
            filename = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):

        model.load_state_dict(torch.load(self.filename))


# def collect_env():
#     """Collect the information of the running environments."""
#     env_info = collect_base_env()
#     return env_info


# def load_dataloader(n_fold, args):
#     work_dir = args.workdir
#     data_root = osp.join(work_dir, 'data')

#     # 选择基因集路径
#     if args.celldataset == 1:
#         celllines_data = osp.join(data_root, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
#     elif args.celldataset == 2:
#         celllines_data = osp.join(data_root, '0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
#     elif args.celldataset == 3:
#         celllines_data = osp.join(data_root, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')

#     # 原始 substructure embedding 路径
#     drugs_data = osp.join(data_root, '1_drug_data/drugSmile_drugSubEmbed_2644.npy')

#     tr_data_items = osp.join(data_root, f'split/{n_fold}_fold_tr_items.npy')
#     val_data_items = osp.join(data_root, f'split/{n_fold}_fold_val_items.npy')
#     test_data_items = osp.join(data_root, f'split/{n_fold}_fold_test_items.npy')

#     # 生成 Dataset 对象
#     tr_dataset_full = MyInMemoryDataset(data_root, tr_data_items, celllines_data, drugs_data, args=args)
#     val_dataset = MyInMemoryDataset(data_root, val_data_items, celllines_data, drugs_data, args=args)
#     test_dataset = MyInMemoryDataset(data_root, test_data_items, celllines_data, drugs_data, args=args)

#     # ---------- SUBSAMPLE START ----------
#     # 仅取训练集的 10% 数据进行实验，保留前 10% 样本
#     total_tr_samples = len(tr_dataset_full)
#     subsample_size = max(1, total_tr_samples // 10)  # 至少 1 个样本
#     # 这里直接取前 subsample_size 个样本，若希望随机抽样，可使用 random.sample
#     tr_indices = list(range(subsample_size))
#     # 使用 torch.utils.data.Subset 创建子集
#     from torch.utils.data import Subset
#     tr_dataset = Subset(tr_dataset_full, tr_indices)
#     # ---------- SUBSAMPLE END ----------

#     #仅取val的10%数据进行实验，保留前10%样本
#     total_val_samples = len(val_dataset)
#     subsample_size = max(1, total_val_samples // 10)  # 至少 1 个样本
#     val_indices = list(range(subsample_size))
#     from torch.utils.data import Subset
#     val_dataset = Subset(val_dataset, val_indices)

#     #仅取test的10%数据进行实验，保留前10%样本
#     total_test_samples = len(test_dataset)
#     subsample_size = max(1, total_test_samples // 10)  # 至少 1 个样本
#     test_indices = list(range(subsample_size))
#     from torch.utils.data import Subset
#     test_dataset = Subset(test_dataset, test_indices)


#     # 创建 DataLoader
#     tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

#     print(f'train data (10% subsample): {len(tr_dataloader)*args.batch_size} samples')
#     print(f'Valid data: {len(val_dataloader)*args.batch_size} samples')
#     print(f'Test data: {len(test_dataloader)*args.batch_size} samples')

#     return tr_dataloader, val_dataloader, test_dataloader

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset
from torch_geometric.data import Data
from dataset.My_inMemory_dataset import MyInMemoryDataset

def load_dataloader(n_fold, args):
    """
    加载训练/验证/测试集的 DataLoader。使用自定义 collate_fn，将一个 batch 的
    Data 对象列表，合并为一个“大” Data，满足模型要求。

    args 中需包含：
      - workdir
      - celldataset (1/2/3)
      - use_1d, use_2d, use_fp, fp_bits
      - batch_size
      - mode ('train' 或 'infer')
    """

    work_dir = args.workdir
    data_root = osp.join(work_dir, 'data')

    # 1. celllines 数据路径
    if args.celldataset == 1:
        celllines_data = osp.join(
            data_root, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(
            data_root, '0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(
            data_root, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')
    else:
        raise ValueError("celldataset must be 1, 2 or 3")

    # 2. 1D 子结构（smiles embedding）数据路径
    drugs_data = osp.join(data_root, '1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    # 3. split 文件
    tr_data_items  = osp.join(data_root, f'split/{n_fold}_fold_tr_items.npy')
    val_data_items = osp.join(data_root, f'split/{n_fold}_fold_val_items.npy')
    test_data_items= osp.join(data_root, f'split/{n_fold}_fold_test_items.npy')

    # 4. 构造 Dataset
    tr_dataset = MyInMemoryDataset(
        data_root=data_root,
        data_items=tr_data_items,
        celllines_data=celllines_data,
        drugs_data=drugs_data,
        dgi_data=None,
        args=args
    )
    val_dataset = MyInMemoryDataset(
        data_root=data_root,
        data_items=val_data_items,
        celllines_data=celllines_data,
        drugs_data=drugs_data,
        dgi_data=None,
        args=args
    )
    test_dataset = MyInMemoryDataset(
        data_root=data_root,
        data_items=test_data_items,
        celllines_data=celllines_data,
        drugs_data=drugs_data,
        dgi_data=None,
        args=args
    )

    # # 仅取训练集的 10% 数据进行实验，保留前 10% 样本
    # total_tr_samples = len(tr_dataset)
    # subsample_size = max(1, total_tr_samples // 10)  # 至少 1 个样本
    # tr_indices = list(range(subsample_size))
    # tr_dataset = Subset(tr_dataset, tr_indices)

    # # 仅取验证集的 10% 数据进行实验
    # total_val_samples = len(val_dataset)
    # subsample_size = max(1, total_val_samples // 10)
    # val_indices = list(range(subsample_size))
    # val_dataset = Subset(val_dataset, val_indices)

    # # 仅取测试集的 10% 数据进行实验
    # total_test_samples = len(test_dataset)
    # subsample_size = max(1, total_test_samples // 10)
    # test_indices = list(range(subsample_size))
    # test_dataset = Subset(test_dataset, test_indices)


    # --------- 自定义 collate_fn 开始 ---------
    def collate_fn(batch_list):
        """
        batch_list: 一个 batch 的 Data 对象列表，
                    Data 里含有（可能）以下字段：
                      - drugA (LongTensor [1, max_d])
                      - drugA_mask (LongTensor [1, max_d])
                      - drugB, drugB_mask
                      - drugA_x (FloatTensor [num_nodes_A, feat_dim])
                      - drugA_edge_index (LongTensor [2, num_edges_A])
                      - drugA_num_nodes (int)
                      - drugA_num_edges (int)
                      - 同理 drugB_x, drugB_edge_index, ...
                      - drugA_fp, drugB_fp
                      - x_cell (FloatTensor [num_genes, num_omics])
                      - dgiA, dgiB (FloatTensor [num_genes])
                      - y (FloatTensor [1])
        我们要把它们拼成一个“批量化”的 Data 对象：
        - drugA: [batch, max_d]
        - drugA_mask: [batch, max_d]
        - drugB, drugB_mask 同理
        - drugA_x: 将所有样本的 node 特征 concat 到一起 → [sum_nodes_A, feat_dim]
        - drugA_edge_index: 所有图的 edge_index concat → [2, sum_edges_A]
        - drugA_num_nodes: Tensor([n0, n1, ..., n_{batch-1}])
        - drugA_num_edges: Tensor([e0, e1, ..., e_{batch-1}])
        - 同理 drugB_x, ...
        - drugA_fp: [batch, n_bits]
        - x_cell: 将所有样本的 x_cell concat → [batch * num_genes, num_omics]
        - dgiA: concat → [batch * num_genes]
        - y: [batch, 1]
        """
        batch_data = Data()

        B = len(batch_list)

        # ----- 1D 子结构 -----
        if args.use_1d == 1:
            # 每个 d.drugA: [1, max_d]，堆叠成 [B, max_d]
            batch_data.drugA = torch.cat([d.drugA for d in batch_list], dim=0)
            batch_data.drugA_mask = torch.cat([d.drugA_mask for d in batch_list], dim=0)
            batch_data.drugB = torch.cat([d.drugB for d in batch_list], dim=0)
            batch_data.drugB_mask = torch.cat([d.drugB_mask for d in batch_list], dim=0)

        # ----- 2D GNN -----
        if args.use_2d == 1:
            # A 分支
            x_list_A = []
            edge_list_A = []
            num_nodes_list_A = []
            num_edges_list_A = []
            for d in batch_list:
                x_list_A.append(d.drugA_x)                   # [n_i, feat_dim]
                edge_list_A.append(d.drugA_edge_index)       # [2, e_i]
                num_nodes_list_A.append(d.drugA_num_nodes)   # int
                num_edges_list_A.append(d.drugA_num_edges)   # int
            # concat
            batch_data.drugA_x = torch.cat(x_list_A, dim=0)                  # [sum n_i, feat_dim]
            batch_data.drugA_edge_index = torch.cat(edge_list_A, dim=1)      # [2, sum e_i]
            batch_data.drugA_num_nodes = torch.tensor(num_nodes_list_A, dtype=torch.long)  # [B]
            batch_data.drugA_num_edges = torch.tensor(num_edges_list_A, dtype=torch.long)  # [B]

            # B 分支同理
            x_list_B = []
            edge_list_B = []
            num_nodes_list_B = []
            num_edges_list_B = []
            for d in batch_list:
                x_list_B.append(d.drugB_x)
                edge_list_B.append(d.drugB_edge_index)
                num_nodes_list_B.append(d.drugB_num_nodes)
                num_edges_list_B.append(d.drugB_num_edges)
            batch_data.drugB_x = torch.cat(x_list_B, dim=0)
            batch_data.drugB_edge_index = torch.cat(edge_list_B, dim=1)
            batch_data.drugB_num_nodes = torch.tensor(num_nodes_list_B, dtype=torch.long)
            batch_data.drugB_num_edges = torch.tensor(num_edges_list_B, dtype=torch.long)

        # ----- 2D Fingerprint -----
        if args.use_fp == 1:
            batch_data.drugA_fp = torch.cat([d.drugA_fp for d in batch_list], dim=0)  # [B, n_bits]
            batch_data.drugB_fp = torch.cat([d.drugB_fp for d in batch_list], dim=0)

        # ----- 细胞 & DGI & 标签 -----
        # 所有 d.x_cell: [num_genes, num_omics]，拼成 [B * num_genes, num_omics]
        batch_data.x_cell = torch.cat([d.x_cell for d in batch_list], dim=0)
        # DGI
        batch_data.dgiA = torch.cat([d.dgiA for d in batch_list], dim=0)
        batch_data.dgiB = torch.cat([d.dgiB for d in batch_list], dim=0)
        # 标签 y: [1]→[B,1]
        batch_data.y = torch.cat([d.y for d in batch_list], dim=0)

        return batch_data
    # --------- 自定义 collate_fn 结束 ---------


    # 5. DataLoader (num_workers 设置为 0，防止内存不足)
    tr_dataloader = TorchDataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_dataloader = TorchDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )
    test_dataloader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )

    print(f"Number of training samples: {len(tr_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return tr_dataloader, val_dataloader, test_dataloader



def load_infer_dataloader(args):

    
    data_root = osp.join(args.workdir,'data')
    if args.celldataset == 1:
        celllines_data = osp.join(data_root,'0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root,'0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root,'0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')
    drugs_data = osp.join(data_root,'1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    data_items = args.infer_path 
    infer_dataset = MyInMemoryDataset(data_root,data_items,celllines_data,drugs_data,args=args)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=4)

    infer_data_arr = np.load(data_items,allow_pickle=True)

    return infer_dataloader,infer_data_arr



def train(model,criterion,opt,dataloader,device,args=None):

    model.train()
    train_loss_sum = 0
    i = 0 
    lr_list = []

    for data in dataloader:
        i += 1
        model.zero_grad()
        x = data.to(device)
        output, _, _ = model(x)
        y = data.y.unsqueeze(1).type(torch.float32).to(device)
        train_loss = criterion(output,y)
        train_loss_sum += train_loss
        train_loss.backward()
        opt.step() 

    train_loss_sum = train_loss_sum.cpu().detach().numpy()       
    loss = train_loss_sum/i
    
    return loss,lr_list
        
 

# def train(model,criterion,opt,dataloader,device,args=None,lr_scheduler=None):

#     model.train()
#     train_loss_sum = 0
#     i = 0 
#     lr_list = []

#     for data in dataloader:
#         i += 1
#         model.zero_grad()
#         x = data.to(device)
#         output, _ = model(x)
        
#         if args.task == 'reg':
#             y = data.y.unsqueeze(1).type(torch.float32).to(device)
#             # y = Scalr(y)
#             train_loss = criterion(output,y)
#         if args.task == 'clf':
#             y = data.y_clf.long().to(device)
#             train_loss = criterion(output,y)

#         train_loss_sum += train_loss
#         # opt.optimizer.zero_grad()
#         # opt.zero_grad()
#         train_loss.backward() 

#         if lr_scheduler == 0:           
#             lr = opt.step()
#             # print(f'{i} batch lr: {lr}') 
#             lr_list.append(lr.cpu().detach().item())
#         else:
#             opt.step()
#             lr = opt.param_groups[0]['lr']
#             lr_list.append(lr)  # 返回实时的学习率

#     train_loss_sum = train_loss_sum.cpu().detach().numpy()       
#     loss = train_loss_sum/i
    
#     return loss,lr_list




def validate(model,criterion,dataloader,device,args=None):

    model.eval()
    y_true = []
    y_pred = []
    i = 0
    
    with torch.no_grad():
        for data in dataloader:
            i += 1
            x = data.to(device) 
            y = data.y.unsqueeze(1).to(device)
            y_true.append(y.view(-1, 1))
            output, _, _= model(x)
            y_pred.append(output)            

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()      
    mse,rmse,mae,r2,pearson,spearman = get_metrics(y_true,y_pred)
    m1,m2,m3,m4,m5,m6,m7 = mse,rmse,mae,r2,pearson,spearman,None

    return m1,m2,m3,m4,m5,m6,m7 



def infer(model,dataloader,device,args=None):

    model.eval()
    y_pred = []
    y_true = []
    cell_embed = []
    attn = []
    i = 0
    
    with torch.no_grad():
        for data in dataloader:
            i += 1
            x = data.to(device)
            output, output_cell_embed, output_attn = model(x)
            if args.output_attn:
                cell_embed.append(output_cell_embed)
                attn.append(output_attn)
            y = data.y.unsqueeze(1).to(device)
            y_true.append(y.view(-1, 1))
            y_pred.append(output)
                    

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()      
    mse,rmse,mae,r2,pearson,spearman = get_metrics(y_true,y_pred)
    print('Infer reslut: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f} '.format(mse,rmse,mae,r2,pearson,spearman))
    
    if cell_embed:
        cell_embed = torch.stack(cell_embed, dim=0).cpu().detach().numpy()
    if attn:
        attn = torch.stack(attn, dim=0).cpu().detach().numpy()

    return y_pred, cell_embed, attn 


