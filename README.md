# GRNPred
AI methods for predicting gene regulatory networks from omics data

How to Run

Run the main script and provide the 3 required input paths:

python main.py \
  --expr_path mESC_Non_specific_CHIP_tf500_ExpressionData.csv \
  --label_path mESC_Non_specific_CHIP_tf500_refNetwork.csv \
  --tf_path mESC_Non_specific_CHIP_tf500_TFs.csv


Make sure the following supporting files are also available in the same working directory (or specify them via optional arguments):

gene_go_pairs.csv

tf_binding_logodds.csv

go_term.csv

text_description.csv
