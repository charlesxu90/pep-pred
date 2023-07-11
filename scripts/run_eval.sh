# Path: scripts/run_eval.sh

# python eval.py --ckpt results/train_smi_bert2/model_final_0.065.pt --config configs/train_smi_bert_test.yaml --clf xgb


# Evaluate by directory
# for ckpt in results/train_smi_bert2/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_smi_bert_test.yaml --clf xgb
# done


for ckpt in results/train_molclip_test/*.pt; do
    echo $ckpt
    python eval.py --ckpt $ckpt --config configs/train_molclip_test.yaml --clf xgb --model_type molclip
done