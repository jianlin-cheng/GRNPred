# GRNPred
AI methods for predicting gene regulatory networks from omics data.

---
## Download Required Supporting Files

```bash
# GO term gene pairs
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/gene_go_pairs.csv

# TF motif binding log-odds
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/tf_binding_logodds.csv

# GO term definitions / names
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/go_term.csv

# Text descriptions for genes
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/text_description.csv
```
Google Drive (backup): https://mailmissouri-my.sharepoint.com/:f:/g/personal/tmnthc_umsystem_edu/EiYQzBD_oypCuVZyJOSY95ABSmPkhgMTEAusQwCFzrDpjQ?e=n0bedS

## How to Run

Run the main script and provide the 3 required input paths.  
The `main.py` will automatically load the data, split edges into 70/15/15 (train/val/test), train the model, and then run final evaluation on the full test split.

If you run out of memory, you can adjust dimension from 128 to 64

```bash
python main.py \
  --expr_path mHSC-GM_Non_specific_CHIP_tf1000_ExpressionData.csv \
  --label_path mHSC-GM_Non_specific_CHIP_tf1000_refNetwork.csv \
  --tf_path    mHSC-GM_Non_specific_CHIP_tf1000_TFs.csv
```

## Required Supporting Files

These must be present in the working directory (or provided via flags like `--priors_path`, `--tf_bind_path`, etc.):

- `gene_go_pairs.csv`
- `tf_binding_logodds.csv`
- `go_term.csv`
- `text_description.csv`


