# Path: scripts/run_eval.sh

# Evaluate by directory

#=== Evaluate pretrained models ===#
# python eval_pretrain.py --ckpt_dir results/train_smi_bert/ --config configs/train_smi_bert_tune.yaml --clf xgb
python eval_pretrain.py --ckpt_dir results/train_smi_bert_tune/ --config configs/train_smi_bert_tune.yaml --clf xgb

# python eval_pretrain.py --ckpt_dir results/train_aa_bert/ --config configs/train_aa_bert_test.yaml --clf rf --model_type aa_bert
# python eval_pretrain.py --ckpt_dir results/train_molclip/ --config configs/train_molclip_test.yaml --clf xgb --model_type molclip

# python eval_pretrain.py --ckpt_dir results/train_pep_bart1/ --config configs/train_pep_bart_test.yaml --clf rf --model_type pep_bart
# python eval_pretrain.py --ckpt_dir results/train_pep_bart-bak/ --config configs/train_pep_bart_test.yaml --clf rf --model_type pep_bart
# python eval_pretrain.py --ckpt_dir results/train_pep_bart_test/ --config configs/train_pep_bart_test.yaml --clf rf --model_type pep_bart

#=== Task-specific evaluation ===#
# python eval_task_finetune.py --ckpt_dir results/CPP924_aa_bert/ --config configs/CPP924_finetune_aa_bert.yaml