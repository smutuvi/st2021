import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from transformers import (
    BertForSequenceClassification,
    RobertaModel,
    AlbertModel,
    BertModel,
    BertForTokenClassification,
    RobertaForSequenceClassification,
    BertForMaskedLM,
    RobertaForMaskedLM,
    AlbertForMaskedLM
    )
from modeling_roberta import RobertaForSequenceClassification_v2
from transformers.modeling_roberta import RobertaLMHead
from transformers.modeling_bert import BertOnlyMLMHead
from transformers.modeling_albert import AlbertMLMHead

PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

PRETRAINED_MODEL_LM_MAP = {
    'bert': BertOnlyMLMHead,
    'roberta': RobertaLMHead,
    'albert': AlbertMLMHead
}

PRETRAINED_MODEL_MAP_SeqClass = {
    'bert': BertForSequenceClassification,
    'roberta': RobertaForSequenceClassification_v2,
    'albert': AlbertModel
}

PRETRAINED_MODEL_MAP_TokenClass = {
    'bert': BertForTokenClassification,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BERT_model(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(BERT_model, self).__init__(bert_config)
        print(args.task_type)
 
        # if args.task_type == 'ner':
        #     self.bert = PRETRAINED_MODEL_MAP_TokenClass[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        # else:
        #     self.bert = PRETRAINED_MODEL_MAP_SeqClass[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        # #self.fc_layer = FCLayer(bert_config.hidden_size, bert_config.num_labels, args.dropout_rate, use_activation=False)
        # #self.lm_head = RobertaLMHead(config = bert_config)
        self.bert = PRETRAINED_MODEL_MAP_TokenClass[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        self.args = args

    def forward(self, input_ids, attention_mask, token_type_ids, inputs_embeds = None, labels = None):
        #print(labels)
        if input_ids is None:
            outputs = self.bert(inputs_embeds = inputs_embeds, attention_mask = attention_mask,
                                token_type_ids = token_type_ids, labels = labels)  # sequence_output, pooled_output, (hidden_states), (attentions)
            #sequence_output = outputs[0]

        elif labels is not None:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels = labels)  # sequence_output, pooled_output, (hidden_states), (attentions)
            #sequence_output = outputs[0]
            #pooled_output = outputs[1]  # [CLS]
            if self.args.task_type == 're' or 'tc':
                '''
                loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
                logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
                '''
                loss, logits = outputs[:2]
            else:
                '''
                loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided)
                Classification loss.
                scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
                '''
                loss, scores = outputs[:2]
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
            #sequence_output = outputs[0]
            #pooled_output = outputs[1]  # [CLS]
            if self.args.task_type == 're' or 'tc':
                logits = outputs[0]
            else:
                scores = outputs[0]
        return outputs
            #logits = self.fc_layer(sequence_output)

    def forward_pretrain(self, input_ids, attention_mask, masked_lm_labels):
        out = self.bert.roberta(input_ids, attention_mask)
        sequence_output = out[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + out[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs