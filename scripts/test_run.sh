#==== Pretrain ====#
# torchrun --nproc_per_node=2 train_bert.py --config configs/train_aa_bert_test.yaml --output_dir results/train_aa_bert_test --debug
# torchrun --nproc_per_node=2 train_bert.py --config configs/train_smi_bert_test.yaml --output_dir results/train_smi_bert_test --debug
torchrun --nproc_per_node=2 train_molclip.py --config configs/train_molclip_test.yaml --output_dir results/train_molclip_test --debug

#==== SMILES BERT finetune ====#
# torchrun --nproc_per_node=2  train_bert.py --config configs/train_smi_bert_test.yaml --output_dir results/train_smi_bert_tune_test --ckpt results/train_smi_bert_test/model_0_0.000.pt
# torchrun --nproc_per_node=2 train_molclip.py --config configs/train_molclip_test.yaml --output_dir results/train_molclip_tune_test --ckpt results/train_molclip_test/model_0_0.000.pt

#==== task specific finetune ====#
# SMILES BERT
# torchrun --nproc_per_node=2  train_bert.py --config configs/train_smi_bert_test.yaml --output_dir results/train_smi_bert_tune_test --ckpt results/train_smi_bert_test/model_0_0.000.pt