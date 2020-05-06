import os
import json
import torch
import random
import numpy as np
from utils.logger import init_logger
from tqdm import tqdm, trange
from utils.utils import acc_f1, loss_acc_plot
from seqeval.metrics import classification_report
from transformers import (AdamW, get_linear_schedule_with_warmup)

logger = init_logger(__name__, os.getcwd())


def fit(model, train_iter, eval_iter, num_train_optimization_steps, label_list, args):

    # ------------------Device----------------------
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # ------------------Optimizer-------------------------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            args.weight_decay
    },
        {
            'params': [
                p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                0.0
        }]

    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # ------------------Initiation & Train-------------------------------------------
    model.to(device)
    global_step = 0

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy
    }
    label_map = {i: label for i, label in enumerate(label_list)}
    if args.do_train:
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            train_loss, train_acc, train_f1 = 0, 0, 0
            eval_loss, eval_acc, eval_f1 = 0, 0, 0
            for step, batch in enumerate(tqdm(train_iter, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask = batch
                print('device of input', input_ids.device)
                print(device)
                print(input_ids)
                bert_out = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, valid_ids=valid_ids, attention_label_mask=label_mask)
                train_loss = model.loss_fn(bert_out, label_ids, label_mask)
                if n_gpu > 1:
                    train_loss = train_loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    train_loss = train_loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                predicts = model.predict(bert_out, label_mask)
                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                label_ids = label_ids.cpu()
                train_acc, train_f1 = acc_f1(predicts, label_ids)
            train_losses.append(train_loss.item())
            train_accuracy.append(train_acc.item())

            if args.do_eval:
                eval_loss, eval_acc, eval_f1 = eval(model, eval_iter, device, args)
                eval_losses.append(eval_loss)
                eval_accuracy.append(eval_acc)

            logger.info(
                '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - train_f1:%4f - eval_f1:%4f\n'
                % (e + 1, train_loss.item(), eval_loss, train_acc, eval_acc, train_f1, eval_f1))

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        model_config = {
            "bert_model": args.bert_model,
            "do_lower": args.do_lower_case,
            "max_seq_length": args.max_seq_length,
            "num_labels": len(label_list),
            "label_map": label_map
        }
        json.dump(
            model_config,
            open(os.path.join(args.output_dir, "model_config.json"), "w"))
        loss_acc_plot(history, args.output_dir)
    else:
        model.to(device)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_iter))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval(model, eval_iter, label_map, device, args, True)


def eval(model, eval_iter_data, device, args, is_report=False):

    model.eval()
    y_true, y_prediction = [], []
    eval_loss, eval_acc = 0, 0
    eval_count = 0
    for step,batch in enumerate(tqdm(eval_iter_data, desc="Evaluation")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask = batch

        with torch.no_grad():
            bert_out = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, valid_ids=valid_ids, attention_label_mask=label_mask)
        eval_los = model.loss_fn(bert_out, label_ids, label_mask)
        eval_loss = eval_los + eval_loss
        eval_count += 1
        predicts = model.predict(bert_out, label_mask)
        y_prediction.append(predicts)

        label_ids = label_ids.view(1, -1)
        label_ids = label_ids[label_ids != -1]
        y_true.append(label_ids)

    eval_predicted = torch.cat(y_prediction, dim=0).cpu()
    eval_labeled = torch.cat(y_true, dim=0).cpu()
    eval_loss = eval_loss.item() / eval_count
    eval_acc, eval_f1 = acc_f1(eval_predicted, eval_labeled)
    if is_report:
        report = classification_report(y_true, y_prediction, digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)

    return eval_loss, eval_acc, eval_f1
