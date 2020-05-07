from transformers import BertPreTrainedModel, BertModel, BertForTokenClassification
from crf import CRF
import torch
from torch import nn


class BertNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels)
        self.layer_weight = nn.Parameter(torch.Tensor(1, config.num_hidden_layers+1))
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                valid_ids=None,
                attention_label_mask=None):
        sequence_output, _, all_hidden_states = self.bert(
            input_ids, attention_mask, token_type_ids)
        batch_size, max_len, feat_dim = sequence_output.shape
        weighted_sequence_output = torch.zeros_like(sequence_output)
        for index, layer_states in enumerate(all_hidden_states):
            weighted_sequence_output += layer_states * self.layer_weight[0][index]
        sequence_output = weighted_sequence_output.to(sequence_output.device)
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        device = sequence_output.device
        valid_output = valid_output.to(device)
        sequence_output = self.dropout(valid_output)
        return self.classifier(sequence_output)

    def loss_fn(self, bert_out, label_ids, label_mask):
        """
        :param bert_out: batch_size * seq_length * num_labels
        :param label_mask: batch_size * seq_length
        :param label_ids: batch_size * seq_length * num_labels
        :return:
        """
        loss = self.crf.negative_log_loss(bert_out, label_ids, label_mask)
        return loss

    def predict(self, bert_out, output_mask):
        predicts = self.crf.get_batch_best_path(bert_out, output_mask)
        return predicts
