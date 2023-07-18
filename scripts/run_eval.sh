# Path: scripts/run_eval.sh

# Evaluate by directory

# python eval.py --ckpt results/train_smi_bert/model_3_0.277.pt --config configs/train_smi_bert_tune.yaml --clf xgb
# for ckpt in results/train_smi_bert/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_smi_bert_tune.yaml --clf xgb
# done

# for ckpt in results/train_smi_bert_tune/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_smi_bert_tune.yaml --clf xgb
# done

# python eval.py --ckpt results/train_aa_bert/model_30_2.413.pt --config configs/train_aa_bert_test.yaml --clf xgb --model_type aa_bert
# for ckpt in results/train_aa_bert/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_aa_bert_test.yaml --clf xgb --model_type aa_bert
# done

# for ckpt in results/train_aa_bert/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_aa_bert_test.yaml --clf rf --model_type aa_bert
# done


# for ckpt in results/train_molclip_test/*.pt; do
#     echo $ckpt
#     python eval.py --ckpt $ckpt --config configs/train_molclip_test.yaml --clf xgb --model_type molclip
# done

for ckpt in results/train_pep_bart/*.pt; do
    echo $ckpt
    python eval.py --ckpt $ckpt --config configs/train_pep_bart.yaml --clf rf --model_type pep_bart
done
