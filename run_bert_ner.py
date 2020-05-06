import argparse
import os
import torch
from utils.logger import init_logger
from transformers import BertConfig, BertTokenizer
from data_prepare import CoNLL2003NerProcessor
from bert_ner_nodel import BertNer
from train import fit
from data_prepare import convert_examples_to_features, prepare_data_loader

logger = init_logger(__name__, os.getcwd())


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.")
    parser.add_argument(
        "--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action='store_true', help="Whether to run eval or not.")
    parser.add_argument(
        "--eval_on",
        default="dev",
        help="Whether to run eval on the dev set or test set.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.")
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Total batch size for eval.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight deay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        '--server_ip',
        type=str,
        default='',
        help="Can be used for distant debugging.")
    parser.add_argument(
        '--server_port',
        type=str,
        default='',
        help="Can be used for distant debugging.")
    args = parser.parse_args()

    # ------------------Parameter Valid Check-------------------------------------------
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processors = {"CoNLL2003_NER": CoNLL2003NerProcessor}
    # ------------------Prepare Data-------------------------------------------
    task_name = args.task_name
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    data_processor = processors[task_name]()
    label_list = data_processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    train_iter = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = data_processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size(
            )
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        train_iter = prepare_data_loader(train_features, args, 'train')

    eval_examples = data_processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples,label_list,args.max_seq_length, tokenizer)
    eval_iter = prepare_data_loader(eval_features,args, 'eval')

    # Prepare model
    if args.do_train:
        config = BertConfig.from_pretrained(
            args.bert_model, num_labels=num_labels, finetuning_task=args.task_name, output_hidden_states=True)
        model = BertNer.from_pretrained(
            args.bert_model,
            config=config)
        fit(model, train_iter, eval_iter, num_train_optimization_steps, label_list, args)
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertNer.from_pretrained(args.output_dir)
        fit(model, train_iter, eval_iter, num_train_optimization_steps, label_list, args)
