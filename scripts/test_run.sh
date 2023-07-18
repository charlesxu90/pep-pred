#===== Encoders =====#
#==== Pretrain ====#
# torchrun --nproc_per_node=2 train_bert.py --config configs/train_aa_bert_test.yaml --output_dir results/train_aa_bert_test --debug
# torchrun --nproc_per_node=2 train_bert.py --config configs/train_smi_bert_test.yaml --output_dir results/train_smi_bert_test --debug

#==== SMILES BERT finetune ====# Improved
# torchrun --nproc_per_node=2  train_bert.py --config configs/train_smi_bert_tune.yaml --output_dir results/train_smi_bert_tune --ckpt results/train_smi_bert/model_3_0.277.pt

#==== task specific finetune ====# Useless
# AA BERT
# torchrun --nproc_per_node=2 finetune_bert.py --config configs/CPP924_finetune_aa_bert.yaml --output_dir results/CPP924_aa_bert --debug --ckpt results/train_aa_bert/model_68_2.269.pt

# SMILES BERT
# torchrun --nproc_per_node=2  finetune_bert.py --config configs/CPP924_finetune_smi_bert.yaml --output_dir results/CPP924_smi_bert --ckpt results/train_smi_bert_tune/model_13_0.003.pt

#===== MolClip =====#
# torchrun --nproc_per_node=2 train_molclip.py --config configs/train_molclip_test.yaml --output_dir results/train_molclip_test --debug --smi_ckpt results/train_smi_bert_tune/model_13_0.003.pt --aa_ckpt results/train_aa_bert/model_68_2.269.pt

#===== Pep BART =====#
torchrun --nproc_per_node=2 train_pbart.py --config configs/train_pep_bart_test.yaml --output_dir results/train_pep_bart_test --debug 