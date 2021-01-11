import os
import csv
import copy
import json
import logging
import random
import torch
from torch.utils.data import TensorDataset

from utils import get_label

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None, true = -1):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.true = true

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class ReInputExample(InputExample):
    def __init__(self, guid, text_a, span_a, span_b, label, true = -1):
        self.guid = guid
        self.text_a = text_a
        self.span_a = span_a
        self.span_b = span_b
        self.label = label
        self.true = true
    
class WiCInputExample(InputExample):
    def __init__(self, guid, text_a, text_b, span_a, span_b, label, true = -1):
        self.guid = guid
        self.text_a = text_a
        self.span_a = span_a
        self.text_b = text_b
        self.span_b = span_b
        self.label = label
        self.true = true


class MaskedLmInstance(object):
    """
    A single set of features of masked data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """
    def __init__(self, input_ids, attention_mask, masked_token_id, masked_true_label,
                 ):
        self.input_ids = input_ids
        self.masked_token_id = masked_token_id
        self.masked_true_label = masked_true_label
        self.attention_mask = attention_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def readfile(fpath):
    f = open(fpath)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')    
        # print(splits[0])
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    # print(data)
    return data
    
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

        
class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def __init__(self, args):
        self.args = args
        # self.num_labels = len(self.get_labels()) + 1
        self.num_label = 11
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}


    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            # if set_type in ['unlabeled']:
            #     label = label
            #     # for x in range(len(sentence)):
            #     #     label[x] = -1
            #     # # label = -1
            # else:
            #     label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)

processors = {
    "epidemic": NerProcessor,
}


def convert_examples_to_features(examples, label_list, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}
    print(label_map)
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label        
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            print(label_1)
            
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]
            labels = labels[0:(max_seq_len - 2)]
            valid = valid[0:(max_seq_len - 2)]
            label_mask = label_mask[0:(max_seq_len - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
                # label_ids.append(int(example.label))
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_len:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(label_ids) == max_seq_len
        assert len(valid) == max_seq_len
        assert len(label_mask) == max_seq_len

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features

def load_and_cache_examples(args, tokenizer, mode):

    processor = processors[args.task](args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            'dist' if args.rule == 1 else 'clean'
        )
    )
    # print(f'HERE: {processor}')
    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        label_list = processor.get_labels()
        features = convert_examples_to_features(examples, label_list, args.max_seq_len, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    size = len(features)
  
 # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)

    #dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset,processor.relation_labels, processor.num_label, processor.id2label, processor.label2id, size

def load_and_cache_unlabeled_examples(args, tokenizer, mode, train_size = 100):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_unlabel_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            'dist' if args.rule == 1 else 'clean'
        )
    )

    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        assert mode == "unlabeled"
        examples = processor.get_examples("unlabeled")
        label_list = processor.get_labels()
        features = convert_examples_to_features(examples, label_list, args.max_seq_len, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

 # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    size = len(features)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)

    #dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, size