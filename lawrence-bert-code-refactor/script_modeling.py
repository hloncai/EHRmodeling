! python ./plot.py \
    --data_dir ./180DaysData_tokens/breast \
    --raw_data_dir ./180DaysData/breast \
    --task breast \
    --bert_model ./pretraining \
    --max_seq_length 512 \
    --train_batch_size 2 \
    --num_day 180  \
    --experiment_number 21 \
    --tokenizer bert-base-uncased \
    --out_directory ./results


! python ./plot.py \
    --data_dir ./180DaysData_tokens/glioma \
    --raw_data_dir ./180DaysData/glioma \
    --task glioma \
    --bert_model ./pretraining \
    --max_seq_length 512 \
    --train_batch_size 2 \
    --num_day 180  \
    --experiment_number 21 \
    --tokenizer bert-base-uncased \
    --out_directory ./results

! python ./plot.py \
    --data_dir ./180DaysData_tokens/lung \
    --raw_data_dir ./180DaysData/lung \
    --task lung \
    --bert_model ./pretraining \
    --max_seq_length 512 \
    --train_batch_size 2 \
    --num_day 180  \
    --experiment_number 21 \
    --tokenizer bert-base-uncased \
    --out_directory ./results

! python ./plot.py \
    --data_dir ./180DaysData_tokens/prostate \
    --raw_data_dir ./180DaysData/prostate \
    --task prostate \
    --bert_model ./pretraining \
    --max_seq_length 512 \
    --train_batch_size 2 \
    --num_day 180  \
    --experiment_number 21 \
    --tokenizer bert-base-uncased \
    --out_directory ./results