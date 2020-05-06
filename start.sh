CURRENT_DIR=`pwd`
export DATA_DIR=$CURRENT_DIR/data

python main.py \
--data_dir=$DATA_DIR \
--bert_model=bert-base-cased \
--task_name=CoNLL2003_NER \
--output_dir=outbase \
--max_seq_length=128 \
--num_train_epochs=50 \
--do_train \
--do_eval \
--use_crf \
--warmup_proportion=0.1 \
--overwrite_output_dir
