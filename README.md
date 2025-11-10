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
## How to Run

Run the main script and provide the 3 required input paths:

```bash
python main.py \
  --expr_path mESC_Non_specific_CHIP_tf500_ExpressionData.csv \
  --label_path mESC_Non_specific_CHIP_tf500_refNetwork.csv \
  --tf_path mESC_Non_specific_CHIP_tf500_TFs.csv
```

## Required Supporting Files

These must be present in the working directory (or provided via flags like `--priors_path`, `--tf_bind_path`, etc.):

- `gene_go_pairs.csv`
- `tf_binding_logodds.csv`
- `go_term.csv`
- `text_description.csv`


