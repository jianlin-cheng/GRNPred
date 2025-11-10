# GRNPred
AI methods for predicting gene regulatory networks from omics data.

---

## How to Run

Run the main script and provide the 3 required input paths:

```bash
python main.py \
  --expr_path mESC_Non_specific_CHIP_tf500_ExpressionData.csv \
  --label_path mESC_Non_specific_CHIP_tf500_refNetwork.csv \
  --tf_path mESC_Non_specific_CHIP_tf500_TFs.csv

Required Supporting Files

Ensure the following files are also available in the same working directory
(or specify their paths with --priors_path / --tf_bind_path / etc.):

gene_go_pairs.csv

tf_binding_logodds.csv

go_term.csv

text_description.csv
