import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from tqdm import tqdm

ATOM_LIST = ['C', 'N', 'O', 'S', 'Cl', 'Br', 'F']

def atom_to_feature_vector(atom):
    """
    Convert an RDKit atom object to a fixed-length feature vector.
    - One-hot over ATOM_LIST (len=7) + 'other' = 8 dims
    - degree (int)
    - total H count (int)
    - is_aromatic (0 or 1)
    """
    symbol = atom.GetSymbol()
    feature = [0] * (len(ATOM_LIST) + 1)
    if symbol in ATOM_LIST:
        feature[ATOM_LIST.index(symbol)] = 1
    else:
        feature[-1] = 1

    feature.append(atom.GetDegree())
    feature.append(atom.GetTotalNumHs())
    feature.append(1 if atom.GetIsAromatic() else 0)

    return torch.tensor(feature, dtype=torch.float32)

def smiles_to_graph(smi):
    """
    Convert a SMILES string to a PyG Data object with:
     - x: node features [num_atoms, feat_dim]
     - edge_index: [2, num_edges] 
    Returns None if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # Build node feature matrix
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_to_feature_vector(atom))
    x = torch.stack(node_feats, dim=0)  # [num_atoms, 11]

    # Build edge index (bidirectional)
    edge_list = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        edge_list.append([a1, a2])
        edge_list.append([a2, a1])
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, num_edges]

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_atoms)

    x_norm = x.norm(p=2, dim=1, keepdim=True)  # [num_atoms,1]
    x_normed = x / (x_norm + 1e-6)

    data = Data(x=x_normed, edge_index=edge_index)
    return data

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate 2D molecular graph dictionary from split SMILES")
    parser.add_argument(
        "--split_dir", type=str, default="data/split",
        help="Directory containing split .npy files"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/1_drug_data/drug_graph_dict.npy",
        help="Output .npy path for graph dict"
    )
    args = parser.parse_args()

    split_files = [os.path.join(args.split_dir, f) for f in os.listdir(args.split_dir) if f.endswith(".npy")]
    all_smiles_set = set()
    for fpath in split_files:
        arr = np.load(fpath, allow_pickle=True)
        for item in arr:
            sA = item[0]
            sB = item[1]
            all_smiles_set.add(sA)
            all_smiles_set.add(sB)
    print(f"Total unique SMILES in splits: {len(all_smiles_set)}")

    drug_graph_dict = {}
    failed = []
    for smi in tqdm(list(all_smiles_set)):
        graph_data = smiles_to_graph(smi)
        if graph_data is None:
            failed.append(smi)
            continue
        node_feat_np = graph_data.x.numpy()
        edge_index_np = graph_data.edge_index.numpy()
        drug_graph_dict[smi] = {
            "node_feat": node_feat_np,
            "edge_index": edge_index_np
        }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, drug_graph_dict)
    print(f"Saved drug graph dict to {args.output_path}. Success: {len(drug_graph_dict)}, Failed: {len(failed)}")
    if failed:
        print("Failed SMILES (graph):")
        for s in failed:
            print("  ", s)

if __name__ == "__main__":
    main()
