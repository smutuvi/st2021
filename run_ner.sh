task=epidemic
MAX_LENGTH=256
MODEL=roberta-base
OUTPUT_DIR=output_dir
BATCH_SIZE=32
NUM_EPOCHS=1
SAVE_STEPS=100
LOGGING_STEPS=100
SEED=42
method=clean
python3 run_ner.py \
  --data_dir=./data \
  --model_type=roberta \
  --labels=./data/labels.txt \
  --model_name_or_path=${MODEL} \
  --output_dir=${OUTPUT_DIR} \
  --max_seq_len=${MAX_LENGTH} \
  --num_train_epochs=${NUM_EPOCHS} \
  --per_gpu_train_batch_size=${BATCH_SIZE} \
  --save_steps=${SAVE_STEPS} \
  --logging_steps=${LOGGING_STEPS} \
	--method=${method} \
  --rule=0 \
  --seed=${SEED} \
  --task=${task} \
	--train_file=train.txt \
	--dev_file=dev.txt \
	--test_file=test.txt \
  --unlabel_file=unlabeled.txt \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \