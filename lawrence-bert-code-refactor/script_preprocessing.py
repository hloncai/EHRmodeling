! python ./preprocessing.py \
    --data_dir ./180DaysData/breast \
    --output_dir ./180DaysData_tokens/breast \
    --task breast \
    --mode all


! python ./preprocessing.py \
    --data_dir ./180DaysData/glioma \
    --output_dir ./180DaysData_tokens/glioma \
    --task glioma \
    --mode all

! python ./preprocessing.py \
    --data_dir ./180DaysData/lung \
    --output_dir ./180DaysData_tokens/lung \
    --task lung \
    --mode all

! python ./preprocessing.py \
    --data_dir ./180DaysData/prostate \
    --output_dir ./180DaysData_tokens/prostate \
    --task prostate \
    --mode all