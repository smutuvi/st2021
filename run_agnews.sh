task=epidemic
gpu=0
method=selftrain
max_seq_len=128
batch_size=32
echo ${method}
python3 main.py \
	--do_train \
	--do_eval \
	--task=${task} \
	--train_file=train.txt \
	--dev_file=dev.txt \
	--test_file=test.txt \
	--unlabel_file=unlabeled.txt \
	--task_type=tc \
	--data_dir="data" \
	--rule=0 \
	--logging_steps=100 \
	--self_train_logging_steps=100 \
	--gpu="${gpu}" \
	--num_train_epochs=3 \
	--weight_decay=1e-4 \
	--method=${method} \
	--batch_size=${batch_size} \
	--max_seq_len=${max_seq_len} \
	--auto_load=1 \
	--self_training_update_period=250 \
	--max_steps=150 \
	--self_training_max_step=2500 \
	--self_training_power=2 \
	--self_training_confreg=0.1 \
	--self_training_contrastive_weight=1 \
	--distmetric='cos' \