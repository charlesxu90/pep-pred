#==== Pretrain ====#

# AA_BERT
# torchrun --nproc_per_node=2 train_bert.py --config configs/train_aa_bert_test.yaml --output_dir results/train_aa_bert_test --debug
# MolClip
# torchrun --nproc_per_node=2 train_molclip.py --config configs/train_molclip_test.yaml --output_dir results/train_molclip_test --debug

#Pep BART
# torchrun --nproc_per_node=2 train_pbart.py --config configs/train_pep_bart_test.yaml --output_dir results/train_pep_bart_test --debug 
torchrun --nproc_per_node=2 train_pbart.py --config configs/train_pep_bart_test.yaml --output_dir results/train_pep_bart_test --debug --aa_ckpt results/train_aa_bert_L40/model_12_2.523.pt

#==== task specific finetune ====#
# AA BERT
# for i in 1..10:
#  do torchrun --nproc_per_node=2 task_finetune.py --config configs/CPP924_aa_bert.yaml --output_dir results/CPP924_aa_bert --debug --ckpt results/train_aa_bert_L40/model_12_2.523.pt;
# done
# torchrun --nproc_per_node=2 task_finetune.py --config configs/CPP924_aa_bert_sia.yaml --output_dir results/CPP924_aa_bert_sia --debug --ckpt results/train_aa_bert_L40/model_12_2.523.pt

# Pep BART
# torchrun --nproc_per_node=2 task_finetune.py --config configs/CPP924_pep_bart.yaml --output_dir results/CPP924_pep_bart --debug --ckpt results/train_pep_bart1/model_13_0.000.pt  --ckpt_model_type pep_bart