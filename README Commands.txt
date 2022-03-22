To generate the needed augmented training data please run these commands:

Guided-adversarial examples:
CUDA_VISIBLE_DEVICES=0 python ./code/BERT/generate_and_save_augmented_examples.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'data/conll2003' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls

CUDA_VISIBLE_DEVICES=6 python ./code/BERT/generate_aug_ex_no_phrase_held_out.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'data/conll2003' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls

CUDA_VISIBLE_DEVICES=3 python ./code/BERT/generate_save_aug_examples_few_shot.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'data/conll2003' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls

TextFlint:
CUDA_VISIBLE_DEVICES=3 python ./code/BERT/generate_examples_for_text_flint.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'data/conll2003' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls

CUDA_VISIBLE_DEVICES=3 python ./code/BERT/generate_examples_for_text_flint_few_shot.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'data/conll2003' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls

Use the 11_NER.ipynb jupyter notebook and the generated jsons for the TextFlint augmenetation patterns: ConcatSent and EntTypos 

----------------------------------------------------------------------------------------

To run all non mixup and non TAVAT models, adjust the augmented-train-percentage parameter (Percentages include  5,10,30,50,100 for 25% held out word phrases, -5,-10,-30,-50,-100 for no held out phrases)
Run these commands for experiments, run the file ending in _few_shot for 5-shot experiments

Regular BERT: --train-mode='regular'
Guided-adversarially trained BERT: --train-mode='rulebasedaug'
CUDA_VISIBLE_DEVICES=6 python ./code/BERT/aug_and_non_aug_train_bert.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'AT' --gpu '6' --labels './data/conll2003/labels.txt' --train-mode='rulebasedaug'  --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --augmented-train-percentage 5

CUDA_VISIBLE_DEVICES=6 python ./code/BERT/aug_and_non_aug_train_bert_few_shot.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'AT' --gpu '6' --labels './data/conll2003/labels.txt' --train-mode='rulebasedaug'  --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --augmented-train-percentage 5



CUDA_VISIBLE_DEVICES=5 python ./code/BERT/augmented_train_dropout_bert_few_shot.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'Dropout5shot100percent' --gpu '6' --labels './data/conll2003/labels.txt' --train-mode='rulebasedaug'  --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --augmented-train-percentage 100

CUDA_VISIBLE_DEVICES=5 python ./code/BERT/augmented_train_dropout_bert.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'Dropout' --gpu '6' --labels './data/conll2003/labels.txt' --train-mode='rulebasedaug'  --max-seq-length 256 --overwrite-output-dir  --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --augmented-train-percentage -100


CUDA_VISIBLE_DEVICES=1 python ./code/BERT/text_flint_aug_train_bert.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'TextFlint' --gpu '6' --labels './data/conll2003/labels.txt' --train-mode='rulebasedaug' --max-seq-length 256 --overwrite-output-dir --do-train  --do-eval --evaluate-during-training  --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --augmented-train-percentage 100

CUDA_VISIBLE_DEVICES=1 python ./code/BERT/text_flint_aug_train_bert_few_shot.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'TextFlint' --gpu '6' --labels './data/conll2003/labels.txt' --train-mode='rulebasedaug' --max-seq-length 256 --overwrite-output-dir --do-train  --do-eval --evaluate-during-training  --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --augmented-train-percentage 100






To run Guided-adversarially trained Mixup BERT model, adjust the augmented-train-percentage parameter (Percentages include  5,10,30,50,100 for 25% held out word phrases, -5,-10,-30,-50,-100 for no held out phrases), 
adjust these parameters as needed: beta-regular alpha-regular alpha-aug beta-aug:

CUDA_VISIBLE_DEVICES=4 python ./code/BERT/augmented_train_mixup_bert.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'Mixup' --gpu '6' --labels './data/conll2003/labels.txt'  --max-seq-length 256 --overwrite-output-dir --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --mix-layers-set 8 9 10  --beta-regular 5 --alpha-regular 150 --alpha-aug 200 --beta-aug 5 --augmented-train-percentage 100

adjust these parameters as needed:  --beta-regular  --alpha-regular  --alpha-aug  --beta-aug  --alpha-outofdomain --beta-outofdomain  --alpha-outofdomain-aug   --beta-outofdomain-aug   for 5-shot:
CUDA_VISIBLE_DEVICES=1 python ./code/BERT/augmented_train_mixup_bert_few_shot.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'mixup' --gpu '6' --labels './data/conll2003/labels.txt'  --max-seq-length 256 --overwrite-output-dir --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --mix-layers-set 8 9 10  --beta-regular 5 --alpha-regular 200 --alpha-aug 150 --beta-aug 5 --alpha-outofdomain 200 --beta-outofdomain 5 --alpha-outofdomain-aug  130 --beta-outofdomain-aug 7  --augmented-train-percentage 30




To run TAVAT model:
CUDA_VISIBLE_DEVICES=4 python ./code/BERT/token_vat.py --model_type bert --model_name_or_path bert-base-cased --do_lower_case --learning_rate 5e-5  --do_train  --do-eval --evaluate-during-training --do_predict --task_name mnli --data_dir 'data/conll2003'  --output_dir SOTATavat --overwrite_output_dir --max_seq_length 256 --save_steps 750 --logging_steps 150 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 0 --num_train_epochs 10 --adv_lr 5e-2 --adv_init_mag 2e-1 --adv_max_norm 5e-1 --adv_steps 2 --vocab_size 28996 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1 --max_steps -1

CUDA_VISIBLE_DEVICES=4 python ./code/BERT/token_vat_few_shot.py --model_type bert --model_name_or_path bert-base-cased --do_lower_case --learning_rate 5e-5  --do_train  --do-eval --evaluate-during-training --do_predict --task_name mnli --data_dir 'data/conll2003'  --output_dir SOTATavat --overwrite_output_dir --max_seq_length 256 --save_steps 750 --logging_steps 150 --evaluate_during_training --per_gpu_train_batch_size 8 --warmup_steps 0 --num_train_epochs 10 --adv_lr 5e-2 --adv_init_mag 2e-1 --adv_max_norm 5e-1 --adv_steps 2 --vocab_size 28996 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1 --max_steps -1


----------------------------------------------------------------------------------------
To evaluate models on out-of-domain:

Non Mixup models:
CUDA_VISIBLE_DEVICES=6 python ./code/BERT/eval_non_mixup_models_OOD.py  --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'mixup' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls

For Mixup model:
CUDA_VISIBLE_DEVICES=6 python ./code/BERT/eval_mixup_model_OOD.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'mixup' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir  --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls



----------------------------------------------------------------------------------------


For ablation study:
To run ablation study of Heuristically augmented trained Mixup BERT model, adjust the augmented-train-percentage parameter (Percentages include  5,10,30,50,100 for 25% held out word phrases), 
and adjust these parameters as needed: beta-regular alpha-regular alpha-aug beta-aug  and run this command:

CUDA_VISIBLE_DEVICES=0 python ./code/BERT/ablation_study.py --data-dir 'data/conll2003' --model-type 'bert'  --model-name 'bert-base-cased' --output-dir 'AblationStudy' --gpu '6' --labels './data/conll2003/labels.txt' --max-seq-length 256 --overwrite-output-dir --do-train  --do-eval --evaluate-during-training --do-predict --batch-size 8 --num-train-epochs 10 --save-steps 750 --seed 1  --eval-batch-size 128 --pad-subtoken-with-real-label --eval-pad-subtoken-with-first-subtoken-only --label-sep-cls --mix-layers-set 8 9 10  --beta-regular 7 --alpha-regular 130 --alpha-aug 200 --beta-aug 5 --augmented-train-percentage 50