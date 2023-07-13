# Path: scripts/run_eval.sh

# Evaluate by directory
for ckpt in results/train_smi_bert/*.pt; do
    echo $ckpt
    python eval.py --ckpt $ckpt --config configs/train_smi_bert_test.yaml --clf xgb
done


# for ckpt in results/train_molclip_test/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_molclip_test.yaml --clf xgb --model_type molclip
# done