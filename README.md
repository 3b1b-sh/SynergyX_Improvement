# SynergyX Optimization: Applying Feature Fusion and reconstructing the cell line data

## Introduction
This project is an optimized version of the original SynergyX model, focusing on two major improvements:
1. Enhanced feature fusion mechanisms for better integration of multi-modal drug representations
2. Improved gene data processing pipeline for more accurate cell line feature extraction

The original SynergyX model demonstrated promising results in drug combination therapy by predicting synergy outcomes. Our optimization aims to further enhance its performance through better feature integration and data processing.

## Model Features
Our optimized version maintains the flexible combinations of three drug encoding methods from the original SynergyX:
1. **1D SMILES Sequence Encoding**: Using Transformer architecture to process drug SMILES sequences
2. **2D Molecular Graph Encoding**: Using GNN to process drug molecular graph structures
3. **2D Molecular Fingerprint Encoding**: Using Morgan molecular fingerprints to represent drug features

Key improvements include:
- Enhanced feature fusion mechanisms for better integration of different drug representations
- Improved gene data processing pipeline using STRING PPI database
- Optimized attention mechanisms for better cross-modal interactions

Users can enable any combination of these encoding methods, and the model will automatically fuse the features from the selected encodings.

## Project Structure
```
.
├── data/                  # Data files and processing scripts
│   ├── 0_cell_data/      # Cell line multi-omics data
│   ├── 1_drug_data/      # Drug-related data
│   ├── split/            # Dataset splits
│   ├── processed/        # Processed drug data (.pt files)
│   ├── raw_data/         # Raw data files
│   │   ├── data/         # STRING PPI data folder
│   │   └── process_gene_data.ipynb # Gene data processing script
│   ├── infer_data/       # Inference data
│   ├── independent dataset/ # Independent test dataset
│   └── data_process.ipynb # Data preprocessing script
├── dataset/              # Dataset related code
├── models/               # Model architecture code
├── saved_model/         # Pre-trained model weights
├── experiment/          # Experiment logs and outputs
├── utils.py            # Utility functions
├── metrics.py          # Evaluation metrics
└── main.py             # Main program entry
```

## Environment Setup
SynergyX is built using PyTorch and PyTorch Geometric. You can create a conda environment using the following commands:

```bash
conda create -n synergyx python=3.8 pip=22.1.2
conda activate synergyx
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.2
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install pandas
pip install subword-nmt
pip install rdkit
pip install biopython  # Required for gene data processing
```

## Data Preparation
1. Download the complete dataset from Zenodo: https://doi.org/10.5281/zenodo.8289295
2. Download STRING PPI data:
   - Visit: https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
   - Extract the downloaded file and place it in `data/raw_data/data/`
3. Process gene data:
   - Navigate to `data/raw_data/process_gene_data.ipynb`
   - Update the file paths in the notebook according to your local setup
   - Run the notebook to process gene data
4. Cell line multi-omics data is stored in `data/0_cell_data/`
5. Drug-related data is stored in `data/1_drug_data/`
6. Dataset splits are stored in `data/split/`
7. Processed drug data (.pt files) will be automatically generated in `data/processed/` during training

## Model Training
### Basic Training Command
```bash
python -u main.py \
    --mode train \
    --nfold 0 \
    --celldataset 2 \
    --device cuda:0 \
    --batch_size 64 \
    --lr 1e-4 \
    --epochs 500 \
    --patience 50 \
    --use_1d 0 \
    --use_2d 0 \
    --use_fp 1 \
    --saved-model ./saved_model/0_fold_fp_only.pth \
    > "./experiment/$(date +'%Y%m%d_%H%M').log"
```

### Encoding Method Selection
Configure different encoding combinations using the following parameters:
- `--use_1d 1/0`: Enable/disable SMILES sequence encoding
- `--use_2d 1/0`: Enable/disable molecular graph encoding
- `--use_fp 1/0`: Enable/disable molecular fingerprint encoding

Examples:
1. SMILES sequence only:
```bash
--use_1d 1 --use_2d 0 --use_fp 0
```

2. Molecular graph only:
```bash
--use_1d 0 --use_2d 1 --use_fp 0
```

3. Molecular fingerprint only:
```bash
--use_1d 0 --use_2d 0 --use_fp 1
```

4. Combine SMILES and molecular graph:
```bash
--use_1d 1 --use_2d 1 --use_fp 0
```

5. Combine all encodings:
```bash
--use_1d 1 --use_2d 1 --use_fp 1
```

## Model Evaluation
Use the following command to evaluate model performance:
```bash
python main.py \
    --mode test \
    --saved-model ./saved_model/0_fold_SynergyX.pt \
    --celldataset 2 \
    --use_1d 1 \
    --use_2d 1 \
    --use_fp 1
```

## Model Prediction
Use the following command for prediction:
```bash
python main.py \
    --mode infer \
    --saved-model ./saved_model/0_fold_SynergyX.pt \
    --infer-path ./data/infer_data/sample_infer_items.npy \
    --output-attn 1 \
    --celldataset 2 \
    --use_1d 1 \
    --use_2d 1 \
    --use_fp 1
```

## Important Notes
1. Ensure at least one encoding method is enabled (`use_1d`, `use_2d`, or `use_fp` must be 1)
2. When using molecular fingerprints, you can specify the number of bits using `--fp_bits` parameter (default: 1024)
3. Prediction results and attention matrices will be saved in the `experiment/` directory
4. During training, processed drug data (.pt files) will be automatically generated in `data/processed/` directory
5. Make sure to update file paths in `process_gene_data.ipynb` according to your local setup

## Reference
This project is an optimized version of the original SynergyX implementation:
- Original Repository: https://github.com/GSanShui/SynergyX
- Original Paper: [SynergyX: a multi-modality mutual attention network for interpretable drug synergy prediction  ]

Key improvements over the original implementation:
1. Enhanced feature fusion mechanisms for better integration of multi-modal drug representations
2. Improved gene data processing pipeline using STRING PPI database
3. Optimized attention mechanisms for better cross-modal interactions
