# GRNPred
AI methods for predicting gene regulatory networks from omics data.

---
## 📦 Download Required Supporting Files

```bash
# Motif score
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/motif_score.tsv

# TF motif binding log-odds
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/tf_binding_logodds.csv

# GO term definitions / names
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/go_term.csv

# Text descriptions for genes
wget https://github.com/tomnguyen4/GRNPred/releases/download/Data/v1.0.0/text_description.csv
```
Google Drive (backup): https://mailmissouri-my.sharepoint.com/:f:/g/personal/tmnthc_umsystem_edu/EiYQzBD_oypCuVZyJOSY95ABSmPkhgMTEAusQwCFzrDpjQ?e=n0bedS

---

## ⚙️ Ensuring Reproducibility

To obtain identical results, please also download the precomputed LLaMA text embedding cache and place it in your project directory:

```text
.cache_llama_embeds/
│
├── all-MiniLM-L6-v2_mESC_Non_specific_CHIP_tf500_ExpressionData.csv_embeds.npy
├── all-MiniLM-L6-v2_mHSC-GM_Celltype_CHIP_tf1000_ExpressionData.csv_embeds.npy
├── ...
```

Google Drive: https://mailmissouri-my.sharepoint.com/:f:/g/personal/tmnthc_umsystem_edu/EttkuWwfXvdOlv0KwyF6c1wB0E5QvugNg9aqJsReTJTIWQ?e=YZzH1s

---

## 🚀 How to Run

Run the main script and provide the 3 required input paths.  
The `main.py` will automatically load the data, split edges into 70/15/15 (train/val/test), train the model, and then run final evaluation on the full test split.

If you run out of memory, you can adjust dimension from 128 to 64. But please keep in mind, the result may not be as good.

```bash
python main.py \
  --expr_path mHSC-GM_Non_specific_CHIP_tf1000_ExpressionData.csv \
  --label_path mHSC-GM_Non_specific_CHIP_tf1000_refNetwork.csv \
  --tf_path    mHSC-GM_Non_specific_CHIP_tf1000_TFs.csv
```
---

## 📑 Required Supporting Files

These must be present in the working directory (or provided via flags like `--priors_path`, `--tf_bind_path`, etc.):

- `gene_go_pairs.csv`
- `tf_binding_logodds.csv`
- `go_term.csv`
- `text_description.csv`


