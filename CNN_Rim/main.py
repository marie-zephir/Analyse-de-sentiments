#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext
import torchtext.datasets as datasets
import model
import train
import mydatasets
from torch.utils.data import Sampler

import random

from transformers import AutoTokenizer

class OnTheFlyRandomIterator(torchtext.data.Iterator):
    def __init__(self, *args, **kwargs):
        super(OnTheFlyRandomIterator, self).__init__(*args, **kwargs)
        self.train_sampler = None  # Set to None to avoid using the default sampler

    def create_batches(self):
        if self.train:
            indices = self.data()
            random.shuffle(indices)
            self.batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        else:
            self.batches = []
            for minibatch in self.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(minibatch, key=self.sort_key))

    def data(self):
        if self.train:
            return [example.text for example in self.dataset.examples]
        else:
            return [example.text for example in self.dataset.examples]


class CustomIterator(torchtext.data.Iterator):
    def random_shuffler(self, data, batch_size):
        indices = list(range(len(data)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            yield indices[i:i + batch_size]

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.07, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load SST dataset
def sst(text_field, label_field, **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    #text_field.vocab = text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = mydatasets.CustomIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size,
                     len(dev_data),
                     len(test_data)),
        **kargs)
    return train_iter, dev_iter, test_iter


# load MR dataset
def mr(text_field, label_field, **kargs):
    csv_path = '../data/csv'
    train_data, dev_data = mydatasets.MyDataset.from_csv(text_field, label_field, csv_path+'/train.csv'), mydatasets.MyDataset.from_csv(text_field, label_field,csv_path + '/dev.csv')
    text_field.build_vocab(train_data.data, dev_data.data)
    #label_field.build_vocab(train_data.data, dev_data.data)
    train_iter = CustomIterator(train_data, batch_size=512)
    dev_iter = CustomIterator(dev_data, batch_size=512)
    print(train_data.data)
    return train_iter, dev_iter


print("load tokenizer distilbert model")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base-sentiment")

print("tokenizer loaded")

# Get the embeddings (hidden states) for the tokens
def preprocessor(batch):
    return tokenizer.tokenize(batch, add_special_tokens=True, padding='max_length')

# load data
print("\nLoading data...")
# Define your fields
TEXT = torchtext.data.Field(batch_first=True,sequential=True, lower=True, tokenize=preprocessor, use_vocab=True)
print("text label set")
LABEL = torchtext.data.LabelField(dtype=torch.int, use_vocab=False)
print("fields ...")
# Specify the path to your CSV file
csv_path = '../data/csv'
print("csv path ...")
# Create an instance of your dataset
#train_ds = mydatasets.MyDataset.from_csv(text_field, label_field, csv_path+'/train.csv')
#val_ds = mydatasets.MyDataset.from_csv(text_field, label_field, csv_path+'/dev.csv')
train_iter, dev_iter = mr(TEXT, LABEL, device=-1, repeat=False)

print("my datasets ...")

# update args and print
args.embed_num = len(TEXT.vocab)
print("embed num ", args.embed_num)
args.class_num = 11
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

# For the test dataset where labels are not present
test_data = mydatasets.MyDataset.test_from_csv(TEXT, csv_path+'/test.csv')

# Create iterator
test_iter = CustomIterator(test_data, batch_size=512)

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, TEXT, LABEL, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))

elif args.test:
    try:
        train.predict(test_iter, cnn, TEXT, LABEL, args.cuda)
    except Exception as e:
        print(e)
        print("\nSorry. The test dataset doesn't exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
