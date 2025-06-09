import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def smiles_to_fingerprint(smi, radius=2, n_bits=1024):
    """
    Convert SMILES into Morgan fingerprint (bit vector) length = n_bits.
    Returns None if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Compute Morgan fingerprint
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for i in range(n_bits):
        arr[i] = int(bitvect.GetBit(i))
    return arr.astype(np.float32)  # convert to float32

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate 2D Morgan fingerprint dict from split SMILES")
    parser.add_argument(
        "--split_dir", type=str, default="data/split",
        help="Directory containing split .npy files (train/val/test), each entry is (drugA_smi, drugB_smi, ...)"
    )
    parser.add_argument(
        "--n_bits", type=int, default=1024,
        help="Fingerprint length (default=1024)"
    )
    parser.add_argument(
        "--radius", type=int, default=2,
        help="Morgan fingerprint radius (default=2)"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output .npy path, e.g. data/1_drug_data/drug_fp_dict_1024bits.npy"
    )
    args = parser.parse_args()

    # 1. Collect all unique SMILES from split .npy files
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

    # 2. Compute fingerprint for each SMILES
    drug_fp_dict = {}
    failed = []
    for smi in tqdm(list(all_smiles_set)):
        fp = smiles_to_fingerprint(smi, radius=args.radius, n_bits=args.n_bits)
        if fp is None:
            failed.append(smi)
            continue
        drug_fp_dict[smi] = fp  # shape [n_bits,], dtype float32

    # 3. Save .npy
    if args.output_path is None:
        output_fn = f"drug_fp_dict_{args.n_bits}bits.npy"
        output_path = os.path.join("data/1_drug_data", output_fn)
    else:
        output_path = args.output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, drug_fp_dict)
    print(f"Saved drug fingerprint dict to {output_path}. Success: {len(drug_fp_dict)}, Failed: {len(failed)}")
    if failed:
        print("Failed to parse these SMILES (fingerprint):")
        for s in failed:
            print(" ", s)

if __name__ == "__main__":
    main()
