# Preprocessing
import sys
import random
import time
import json
import os
import argparse
from utils import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from thop import profile
# Modeling
import torch
from model import BiLSTM
from model import PGD_contrastive
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras import backend as K

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dataset", type=str, choices=["CLINC", "CLINC_OOD"], required=True,
                        help="The dataset to use, ATIS or SNIPS.")
    parser.add_argument("--proportion", type=int, required=True,
                        help="The proportion of seen classes, range from 0 to 100.")
    parser.add_argument("--seen_classes", type=str, nargs="+", default=None,
                        help="The specific seen classes.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both", "find_threshold"], default="both",
                        help="Specify running mode: only train, only test or both.")
    parser.add_argument("--setting", type=str, nargs="+", default=None,
                        help="The settings to detect ood samples, e.g. 'lof' or 'gda_lsqr")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory contains model file (.h5), requried when test only.")
    parser.add_argument("--seen_classes_seed", type=int, default=None,
                        help="The random seed to randomly choose seen classes.")
    # default arguments
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use GPU or not.")
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="The gpu device to use.")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="The directory to store training models & logs.")
    parser.add_argument("--experiment_No", type=str, default="",
                        help="Manually setting of experiment number.")
    # model hyperparameters
    parser.add_argument("--embedding_file", type=str,
                        default="./glove_embeddings/glove.6B.300d.txt",
                        help="The embedding file to use.")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="The dimension of hidden state.")
    parser.add_argument("--contractive_dim", type=int, default=32,
                        help="The dimension of hidden state.")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="The dimension of word embeddings.")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="The max sequence length. When set to None, it will be implied from data.")
    parser.add_argument("--max_num_words", type=int, default=10000,
                        help="The max number of words.")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="The layers number of lstm.")
    parser.add_argument("--do_normalization", type=bool, default=True,
                        help="whether to do normalization or not.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="relative weights of classified loss.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="relative weights of adversarial classified loss.")
    parser.add_argument("--unseen_proportion", type=int, default=100,
                        help="proportion of unseen class examples to add in, range from 0 to 100.")
    parser.add_argument("--mask_proportion", type=int, default=0,
                        help="proportion of seen class examples to mask, range from 0 to 100.")
    parser.add_argument("--ood_loss", action="store_true",
                        help="whether ood examples to backpropagate loss or not.")
    parser.add_argument("--adv", action="store_true",
                        help="whether to generate perturbation through adversarial attack.")
    parser.add_argument("--cont_loss", action="store_true",
                        help="whether to backpropagate contractive loss or not.")
    parser.add_argument("--norm_coef", type=float, default=0.1,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--n_plus_1", action="store_true",
                        help="treat out of distribution examples as the N+1 th class")
    parser.add_argument("--augment", action="store_true",
                        help="whether to use back translation to enhance the ood data")
    parser.add_argument("--cl_mode", type=int, default=1,
                        help="mode for computing contrastive loss")
    parser.add_argument("--lmcl", action="store_true",
                        help="whether to use LMCL loss")
    parser.add_argument("--cont_proportion", type=float, default=1.0,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--dataset_proportion", type=float, default=100,
                        help="proportion for each in-domain data")
    parser.add_argument("--use_bert", action="store_true",
                        help="whether to use bert")
    parser.add_argument("--sup_cont", action="store_true",
                        help="whether to add supervised contrastive loss")
    # training hyperparameters
    parser.add_argument("--ind_pre_epoches", type=int, default=10,
                        help="Max epoches when in-domain pre-training.")
    parser.add_argument("--supcont_pre_epoches", type=int, default=100,
                        help="Max epoches when in-domain supervised contrastive pre-training.")
    parser.add_argument("--aug_pre_epoches", type=int, default=100,
                        help="Max epoches when adversarial contrastive training.")
    parser.add_argument("--finetune_epoches", type=int, default=20,
                        help="Max epoches when finetune model")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience when applying early stop.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Mini-batch size for train and validation")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="weight_decay")
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    args = parser.parse_args()
    return args


args = parse_args()
dataset = args.dataset
proportion = args.proportion
BETA = args.beta
ALPHA = args.alpha
DO_NORM = args.do_normalization
NUM_LAYERS = args.num_layers
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = args.batch_size
EMBEDDING_FILE = args.embedding_file
MAX_SEQ_LEN = args.max_seq_len
MAX_NUM_WORDS = args.max_num_words
EMBEDDING_DIM = args.embedding_dim
CON_DIM = args.contractive_dim
OOD_LOSS = args.ood_loss
CONT_LOSS = args.cont_loss
ADV = args.adv
NORM_COEF = args.norm_coef
LMCL = args.lmcl
CL_MODE = args.cl_mode
USE_BERT = args.use_bert
SUP_CONT = args.sup_cont
CUDA = args.cuda
df, partition_to_n_row = load_data(dataset)

df['content_words'] = df['text'].apply(lambda s: word_tokenize(s))
texts = df['content_words'].apply(lambda l: " ".join(l))

# Do not filter out "," and "."
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<UNK>", filters='!"#$%&()*+-/:;<=>@[\]^_`{|}~')

tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
sequences_pad = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Train-valid-test split
idx_train = (None, partition_to_n_row['train'])
idx_valid = (partition_to_n_row['train'], partition_to_n_row['train'] + partition_to_n_row['valid'])
idx_test = (partition_to_n_row['train'] + partition_to_n_row['valid'], partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'])
idx_cont = (partition_to_n_row['train'] + partition_to_n_row['valid'] + partition_to_n_row['test'], None)

X_train = sequences_pad[idx_train[0]:idx_train[1]]
X_valid = sequences_pad[idx_valid[0]:idx_valid[1]]
X_test = sequences_pad[idx_test[0]:idx_test[1]]
X_cont = sequences_pad[idx_cont[0]:idx_cont[1]]

df_train = df[idx_train[0]:idx_train[1]]
df_valid = df[idx_valid[0]:idx_valid[1]]
df_test = df[idx_test[0]:idx_test[1]]
df_cont = df[idx_cont[0]:idx_cont[1]]

y_train = df_train.label.reset_index(drop=True)
y_valid = df_valid.label.reset_index(drop=True)
y_test = df_test.label.reset_index(drop=True)
y_cont = df_cont.label.reset_index(drop=True)
train_text = df_train.text.reset_index(drop=True)
valid_text = df_valid.text.reset_index(drop=True)
test_text = df_test.text.reset_index(drop=True)
cont_text = df_cont.text.reset_index(drop=True)
print("cont: %d" % (X_cont.shape[0]))

n_class = y_train.unique().shape[0]
if 'CLINC_OOD' in args.dataset and not args.n_plus_1:
    n_class -= 1
if args.augment and 'oos_b' in list(y_train.unique()):
    n_class -= 1
n_class_seen = round(n_class * proportion / 100)
print(n_class_seen)

if args.seen_classes is None:
    if args.seen_classes_seed is not None:
        random.seed(args.seen_classes_seed)
        y_cols = y_train.unique()
        y_cols_lst = list(y_cols)
        if 'oos' in y_cols_lst:
            y_cols_lst.remove('oos')
        if 'oos_b' in y_cols_lst:
            y_cols_lst.remove('oos_b')
        random.shuffle(y_cols_lst)
        y_cols_seen = y_cols_lst[:n_class_seen]
        y_cols_unseen = y_cols_lst[n_class_seen:]
    else:
        # Original implementation
        weighted_random_sampling = False
        if weighted_random_sampling:
            y_cols = y_train.unique()
            y_vc = y_train.value_counts()
            y_vc = y_vc / y_vc.sum()
            y_cols_seen = np.random.choice(y_vc.index, n_class_seen, p=y_vc.values, replace=False)
            y_cols_unseen = [y_col for y_col in y_cols if y_col not in y_cols_seen]
        else:
            y_cols = list(y_train.unique())
            if 'oos' in y_cols and not args.n_plus_1:
                y_cols.remove('oos')
            if args.augment and 'oos_b' in y_cols:
                y_cols.remove('oos_b')
            y_cols_seen = random.sample(y_cols, n_class_seen)
            y_cols_unseen = [y_col for y_col in y_cols if y_col not in y_cols_seen]
else:
    y_cols = y_train.unique()
    y_cols_seen = [y_col for y_col in y_cols if y_col in args.seen_classes and y_col != 'oos']
    y_cols_unseen = [y_col for y_col in y_cols if y_col not in args.seen_classes]
print(y_cols_seen)
print(y_cols_unseen)

y_cols_unseen_b = []
if 'CLINC_OOD' in args.dataset and not args.n_plus_1:
    y_cols_unseen = ['oos']
if args.augment:
    y_cols_unseen = ['oos']
    y_cols_unseen_b = ['oos_b']

for i in range(len(y_cols_seen)):
    tmp_idx = y_train[y_train.isin([y_cols_seen[i]])]
    tmp_idx = tmp_idx[:int(args.dataset_proportion / 100 * len(tmp_idx))].index
    if not i:
        part_train_seen_idx = tmp_idx
    else:
        part_train_seen_idx = np.concatenate((part_train_seen_idx, tmp_idx), axis=0)

train_seen_idx = y_train[y_train.isin(y_cols_seen)].index
train_ood_idx = y_train[y_train.isin(y_cols_unseen)]
train_ood_idx = train_ood_idx[:int(args.unseen_proportion / 100 * len(train_ood_idx))].index

valid_seen_idx = y_valid[y_valid.isin(y_cols_seen)].index
valid_ood_idx = y_valid[y_valid.isin(y_cols_unseen)]
valid_ood_idx = valid_ood_idx[:int(args.unseen_proportion / 100 * len(valid_ood_idx))].index

test_seen_idx = y_test[y_test.isin(y_cols_seen)].index
test_ood_idx = y_test[y_test.isin(y_cols_unseen)].index

src_cols = ['src']
bt_cols = ['bt']
src_idx = y_cont[y_cont.isin(src_cols)]
ind_src_idx = src_idx[:int(args.cont_proportion * 0.8 * len(src_idx))].index
ood_src_idx = src_idx[int(0.8 * len(src_idx)):int(0.8 * len(src_idx) + args.cont_proportion * 0.2 * len(src_idx))].index
bt_idx = y_cont[y_cont.isin(bt_cols)]
ind_bt_idx = bt_idx[:int(args.cont_proportion * 0.8 * len(bt_idx))].index
ood_bt_idx = bt_idx[int(0.8 * len(bt_idx)):int(0.8 * len(bt_idx) + args.cont_proportion * 0.2 * len(bt_idx))].index

X_train_seen = X_train[part_train_seen_idx]
X_train_ood = X_train[train_ood_idx]
y_train_seen = y_train[part_train_seen_idx]
train_seen_text = list(train_text[part_train_seen_idx])
train_unseen_text = list(train_text[train_ood_idx])
X_valid_seen = X_valid[valid_seen_idx]
X_valid_ood = X_valid[valid_ood_idx]
y_valid_seen = y_valid[valid_seen_idx]
valid_seen_text = list(valid_text[valid_seen_idx])
valid_unseen_text = list(valid_text[valid_ood_idx])
X_test_seen = X_test[test_seen_idx]
X_test_ood = X_test[test_ood_idx]
y_test_seen = y_test[test_seen_idx]
test_seen_text = list(test_text[test_seen_idx])
test_unseen_text = list(test_text[test_ood_idx])

print("train : valid : test = %d : %d : %d" % (X_train_seen.shape[0], X_valid_seen.shape[0], X_test_seen.shape[0]))

src_ind_x = X_cont[ind_src_idx]
src_ind_y = y_cont[ind_src_idx]
bt_ind_x = X_cont[ind_bt_idx]
bt_ind_y = y_cont[ind_bt_idx]
src_ood_x = X_cont[ood_src_idx]
src_ood_y = y_cont[ood_src_idx]
bt_ood_x = X_cont[ood_bt_idx]
bt_ood_y = y_cont[ood_bt_idx]

if y_cols_unseen_b:
    train_ood_idx_b = y_train[y_train.isin(y_cols_unseen_b)].index
    X_train_ood_b = X_train[train_ood_idx_b]

le = LabelEncoder()
le.fit(y_train_seen)
y_train_idx = le.transform(y_train_seen)
y_valid_idx = le.transform(y_valid_seen)
y_test_idx = le.transform(y_test_seen)
ood_index = y_test_idx[0]
y_train_onehot = to_categorical(y_train_idx)
y_valid_onehot = to_categorical(y_valid_idx)
y_test_onehot = to_categorical(y_test_idx)
for i in range(int(args.mask_proportion / 100 * len(y_train_onehot))):
    y_train_onehot[i] = [0.0] * n_class_seen
for i in range(int(args.mask_proportion / 100 * len(y_valid_onehot))):
    y_valid_onehot[i] = [0.0] * n_class_seen
y_train_ood = np.array([[0.0] * n_class_seen for _ in range(len(train_ood_idx))])
y_valid_ood = np.array([[0.0] * n_class_seen for _ in range(len(valid_ood_idx))])
y_test_ood = np.array([[0.0] * n_class_seen for _ in range(len(test_ood_idx))])

y_test_mask = y_test.copy()
y_test_mask[y_test_mask.isin(y_cols_unseen)] = 'unseen'
train_text = train_seen_text + train_unseen_text
valid_text = valid_seen_text + valid_unseen_text
test_text = list(test_text)
if not args.unseen_proportion:
    train_data_raw = train_data = (X_train_seen, y_train_onehot)
    valid_data_raw = valid_data = (X_valid_seen, y_valid_onehot)
else:
    train_data_raw = (X_train_seen, y_train_onehot)
    valid_data_raw = (X_valid_seen, y_valid_onehot)
    train_data_ood = (X_train_ood, y_train_ood)
    valid_data_ood = (X_valid_ood, y_valid_ood)
    train_data = (np.concatenate((X_train_seen,X_train_ood),axis=0), np.concatenate((y_train_onehot,y_train_ood),axis=0))
    valid_data = (np.concatenate((X_valid_seen,X_valid_ood),axis=0), np.concatenate((y_valid_onehot,y_valid_ood),axis=0))
test_data = (X_test, y_test_mask)
test_data_4np1 = (X_test, y_test_onehot)
if args.augment:
    train_augment = (np.concatenate((src_ind_x,src_ood_x),axis=0),np.concatenate((bt_ind_x,bt_ood_x),axis=0))


class DataLoader(object):
    def __init__(self, data, batch_size, mode='train', use_bert=False, raw_text=None):
        self.use_bert = use_bert
        if self.use_bert:
            self.inp = list(raw_text)
        else:
            self.inp = data[0]
        self.tgt = data[1]
        self.batch_size = batch_size
        self.n_samples = len(data[0])
        self.n_batches = self.n_samples // self.batch_size
        self.mode = mode
        self._shuffle_indices()

    def _shuffle_indices(self):
        if self.mode == 'test':
            self.indices = np.arange(self.n_samples)
        else:
            self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append((self.inp[_index],self.tgt[_index]))
            self.index += 1
            n += 1
        self.batch_index += 1
        seq, label = tuple(zip(*batch))
        if not self.use_bert:
            seq = torch.LongTensor(seq)
        if self.mode not in ['test','augment']:
            label = torch.FloatTensor(label)
        elif self.mode == 'augment':
            label = torch.LongTensor(label)

        return seq, label

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

if args.mode in ["train", "both"]:
    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_allow_growth(device=args.gpu_device)

    timestamp = str(time.time())  # strftime("%m%d%H%M")
    if args.experiment_No:
        output_dir = os.path.join(args.output_dir, f"{dataset}-{proportion}-{args.experiment_No}")
    else:
        output_dir = os.path.join(args.output_dir, f"{dataset}-{proportion}-{timestamp}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, "seen_classes.txt"), "w") as f_out:
        f_out.write("\n".join(le.classes_))
    with open(os.path.join(output_dir, "unseen_classes.txt"), "w") as f_out:
        f_out.write("\n".join(y_cols_unseen))

    if not USE_BERT:
        print("Load pre-trained GloVe embedding...")
        MAX_FEATURES = min(MAX_NUM_WORDS, len(word_index)) + 1  # +1 for PAD
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (MAX_FEATURES, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_FEATURES: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix = None

    filepath = os.path.join(output_dir, 'model_best.pkl')
    model = BiLSTM(embedding_matrix, BATCH_SIZE, HIDDEN_DIM, CON_DIM, NUM_LAYERS, n_class_seen, DO_NORM, ALPHA, BETA, OOD_LOSS, ADV, CONT_LOSS, NORM_COEF, CL_MODE, LMCL, use_bert=USE_BERT, sup_cont=SUP_CONT, use_cuda=CUDA)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.cuda()

    #in-domain pre-training
    best_f1 = 0

    if args.sup_cont:
        for epoch in range(1,args.supcont_pre_epoches+1):
            global_step = 0
            losses = []
            train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=train_seen_text)
            train_iterator = tqdm(
                train_loader, initial=global_step,
                desc="Iter (loss=X.XXX)")
            model.train()
            for j, (seq, label) in enumerate(train_iterator):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                loss = model(seq, None, label, mode='ind_pre')
                train_iterator.set_description('Iter (sup_cont_loss=%5.3f)' % (loss.item()))
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                global_step += 1
            print('Epoch: [{0}] :  Loss {loss:.4f}'.format(
                epoch, loss=sum(losses)/global_step))
            torch.save(model, filepath)

    for epoch in range(1,args.ind_pre_epoches+1):
        global_step = 0
        losses = []
        train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=train_seen_text)
        train_iterator = tqdm(
            train_loader, initial=global_step,
            desc="Iter (loss=X.XXX)")
        valid_loader = DataLoader(valid_data, BATCH_SIZE, use_bert=USE_BERT, raw_text=valid_text)
        model.train()
        for j, (seq, label) in enumerate(train_iterator):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            if epoch == 1:
                loss = model(seq, None, label, mode='finetune')
            else:
                loss = model(seq, None, label, sim=sim, mode='finetune')
            train_iterator.set_description('Iter (ce_loss=%5.3f)' % (loss.item()))
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            global_step += 1
        print('Epoch: [{0}] :  Loss {loss:.4f}'.format(
            epoch, loss=sum(losses)/global_step))

        model.eval()
        predict = []
        target = []
        if args.cuda:
            sim = torch.zeros((n_class_seen, HIDDEN_DIM*2)).cuda()
        else:
            sim = torch.zeros((n_class_seen, HIDDEN_DIM * 2))
        for j, (seq, label) in enumerate(valid_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            output = model(seq, None, label, mode='validation')
            predict += output[0]
            target += output[1]
            sim += torch.mm(label.T, output[2])
        sim = sim / len(predict)
        n_sim = sim.norm(p=2, dim=1, keepdim=True)
        sim = (sim @ sim.t()) / (n_sim * n_sim.t()).clamp(min=1e-8)
        if args.cuda:
            sim = sim - 1e4 * torch.eye(n_class_seen).cuda()
        else:
            sim = sim - 1e4 * torch.eye(n_class_seen)
        sim = torch.softmax(sim, dim=1)
        f1 = metrics.f1_score(target, predict, average='macro')
        if f1 > best_f1:
            torch.save(model, filepath)
            best_f1 = f1
        print('f1:{f1:.4f}'.format(f1=f1))


if args.mode in ["test", "both", "find_threshold"]:

    if args.n_plus_1:
        test_loader = DataLoader(test_data_4np1, BATCH_SIZE, use_bert=USE_BERT)
        torch.no_grad()
        model.eval()
        predict = []
        target = []
        for j, (seq, label) in enumerate(test_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            output = model(seq, label, 'valid')
            predict += output[1]
            target += output[0]
        m = np.zeros((len(y_cols_seen),len(y_cols_seen)))
        for i in range(len(predict)):
            m[target[i]][predict[i]] += 1
        m[[ood_index, len(y_cols_seen) - 1], :] = m[[len(y_cols_seen) - 1, ood_index], :]
        m[:, [ood_index, len(y_cols_seen) - 1]] = m[:, [len(y_cols_seen) - 1, ood_index]]
        print(get_score(m))


    else:
        if args.mode in ["test","find_threshold"]:
            model_dir = args.model_dir
        else:
            model_dir = output_dir
        if args.cuda:
            model = torch.load(os.path.join(model_dir, "model_best.pkl"), map_location='cuda:0')
        else:
            model = torch.load(os.path.join(model_dir, "model_best.pkl"), map_location='cpu')
        train_loader = DataLoader(train_data_raw, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=train_seen_text)
        valid_loader = DataLoader(valid_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=valid_seen_text)
        valid_ood_loader = DataLoader(valid_data_ood, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=valid_unseen_text)
        test_loader = DataLoader(test_data, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=test_text)
        torch.no_grad()
        model.eval()
        predict = []
        target = []
        for j, (seq, label) in enumerate(valid_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            output = model(seq, None, label, mode='validation')
            predict += output[1]
            target += output[0]
        f1 = metrics.f1_score(target, predict, average='macro')
        print(f"in-domain f1:{f1}")

        valid_loader = DataLoader(valid_data_raw, BATCH_SIZE, 'test', use_bert=USE_BERT, raw_text=valid_seen_text)
        classes = list(le.classes_) + ['unseen']
        #print(list(le.classes_))
        #classes = list(le.classes_)
        feature_train = None
        feature_valid = None
        feature_valid_ood = None
        feature_test = None
        prob_train = None
        prob_valid = None
        prob_valid_ood = None
        prob_test = None
        for j, (seq, label) in enumerate(train_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
            output = model(seq, None, None, mode='test')
            if feature_train != None:
                feature_train = torch.cat((feature_train,output[1]),dim=0)
                prob_train = torch.cat((prob_train,output[0]),dim=0)
            else:
                feature_train = output[1]
                prob_train = output[0]
        for j, (seq, label) in enumerate(valid_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
            output = model(seq, None, None, mode='test')
            if feature_valid != None:
                feature_valid = torch.cat((feature_valid,output[1]),dim=0)
                prob_valid = torch.cat((prob_valid,output[0]),dim=0)
            else:
                feature_valid = output[1]
                prob_valid = output[0]
        for j, (seq, label) in enumerate(valid_ood_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
            output = model(seq, None, None, mode='test')
            if feature_valid_ood != None:
                feature_valid_ood = torch.cat((feature_valid_ood,output[1]),dim=0)
                prob_valid_ood = torch.cat((prob_valid_ood,output[0]),dim=0)
            else:
                feature_valid_ood = output[1]
                prob_valid_ood = output[0]
        for j, (seq, label) in enumerate(test_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
            output = model(seq, None, None, mode='test')
            if feature_test != None:
                feature_test = torch.cat((feature_test,output[1]),dim=0)
                prob_test = torch.cat((prob_test, output[0]), dim=0)
            else:
                feature_test = output[1]
                prob_test = output[0]
        feature_train = feature_train.cpu().detach().numpy()
        feature_valid = feature_valid.cpu().detach().numpy()
        feature_valid_ood = feature_valid_ood.cpu().detach().numpy()
        feature_test = feature_test.cpu().detach().numpy()
        prob_train = prob_train.cpu().detach().numpy()
        prob_valid = prob_valid.cpu().detach().numpy()
        prob_valid_ood = prob_valid_ood.cpu().detach().numpy()
        prob_test = prob_test.cpu().detach().numpy()
        if args.mode == 'find_threshold':
            settings = ['gda_lsqr_'+str(10.0+1.0*(i)) for i in range(20)]
        else:
            settings = args.setting
        for setting in settings:
            pred_dir = os.path.join(model_dir, f"{setting}")
            if not os.path.exists(pred_dir):
                os.mkdir(pred_dir)
            setting_fields = setting.split("_")
            ood_method = setting_fields[0]

            assert ood_method in ("lof", "gda", "msp")

            if ood_method == "lof":
                method = 'LOF (LMCL)'
                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
                lof.fit(feature_train)
                l = len(feature_test)
                y_pred_lof = pd.Series(lof.predict(feature_test))
                test_info = get_test_info(texts=texts[idx_test[0]:idx_test[0]+l],
                                          label=y_test[:l],
                                          label_mask=y_test_mask[:l],
                                          softmax_prob=prob_test,
                                          softmax_classes=list(le.classes_),
                                          lof_result=y_pred_lof,
                                          save_to_file=True,
                                          output_dir=pred_dir)
                pca_visualization(feature_test, y_test_mask[:l], classes, os.path.join(pred_dir, "pca_test.png"))
                df_seen = pd.DataFrame(prob_test, columns=le.classes_)
                df_seen['unseen'] = 0

                y_pred = df_seen.idxmax(axis=1)
                y_pred[y_pred_lof[y_pred_lof == -1].index] = 'unseen'
                cm = confusion_matrix(y_test_mask[:l], y_pred, classes)

                f, f_seen, f_unseen, p_unseen, r_unseen = get_score(cm)
                plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                                      title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                print(cm)
                log_pred_results(f, f_seen, f_unseen, p_unseen, r_unseen, classes, pred_dir, cm, OOD_LOSS, ADV, CONT_LOSS)
            elif ood_method == "gda":
                solver = setting_fields[1] if len(setting_fields) > 1 else "lsqr"
                threshold = setting_fields[2] if len(setting_fields) > 2 else "auto"
                distance_type = setting_fields[3] if len(setting_fields) > 3 else "mahalanobis"
                assert solver in ("svd", "lsqr")
                assert distance_type in ("mahalanobis", "euclidean")
                l = len(feature_test)
                method = 'GDA (LMCL)'
                gda = LinearDiscriminantAnalysis(solver=solver, shrinkage=None, store_covariance=True)
                gda.fit(prob_train, y_train_seen[:len(prob_train)])
                # print(np.max(gda.covariance_class.diagonal()))
                # print(np.min(gda.covariance_class.diagonal()))
                # print(np.mean(gda.covariance_class.diagonal()))
                # print(np.median(gda.covariance_class.diagonal()))
                # print(np.max(np.linalg.norm(gda.covariance_, axis=0)))
                # print(np.min(np.linalg.norm(gda.covariance_, axis=0)))
                # print(np.mean(np.linalg.norm(gda.covariance_, axis=0)))
                # print(np.median(np.linalg.norm(gda.covariance_, axis=0)))
                # dis_matrix = np.matmul(gda.means_, gda.means_.T)
                # K = [1,5,10,30,50]
                # for k in K:
                #     knn = naive_arg_topK(dis_matrix, k, axis=1)
                #     sum = 0
                #     for i in range(knn.shape[0]):
                #         for j in knn[i]:
                #             sum += dis_matrix[i][j]
                #     print(sum/(k*knn.shape[0]))
                if threshold == "auto":
                    # feature_valid_seen = get_deep_feature.predict(valid_data[0])
                    # valid_unseen_idx = y_valid[~y_valid.isin(y_cols_seen)].index
                    # feature_valid_ood = get_deep_feature.predict(X_valid[valid_unseen_idx])
                    seen_m_dist = confidence(prob_valid, gda.means_, distance_type, gda.covariance_).min(axis=1)
                    unseen_m_dist = confidence(prob_valid_ood, gda.means_, distance_type, gda.covariance_).min(axis=1)
                    threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist)
                    # seen_m_dist = confidence(feature_valid, gda.means_, distance_type, gda.covariance_).min(axis=1)
                    # unseen_m_dist = confidence(feature_valid_ood, gda.means_, distance_type, gda.covariance_).min(axis=1)
                    # threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist)
                else:
                    threshold = float(threshold)

                y_pred = pd.Series(gda.predict(prob_test))
                gda_result = confidence(prob_test, gda.means_, distance_type, gda.covariance_)
                test_info = get_test_info(texts=texts[idx_test[0]:idx_test[0]+l],
                                          label=y_test[:l],
                                          label_mask=y_test_mask[:l],
                                          softmax_prob=prob_test,
                                          softmax_classes=list(le.classes_),
                                          gda_result=gda_result,
                                          gda_classes=gda.classes_,
                                          save_to_file=True,
                                          output_dir=pred_dir)
                #pca_visualization(prob_test, y_test_mask[:l], classes, os.path.join(pred_dir, "pca_test.png"))
                #pca_visualization(prob_train, y_train[:15000], classes, os.path.join(pred_dir, "pca_test.png"))
                #pca_visualization(feature_test, y_test_mask[:l], classes, os.path.join(pred_dir, "pca_test.png"))
                y_pred_score = pd.Series(gda_result.min(axis=1))
                y_pred[y_pred_score[y_pred_score > threshold].index] = 'unseen'
                cm = confusion_matrix(y_test_mask[:l], y_pred, classes)
                f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                # plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                #                       title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                print(cm)
                #log_pred_results(f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen, classes, pred_dir, cm, OOD_LOSS, ADV, CONT_LOSS, threshold)
            elif ood_method == "msp":
                threshold = setting_fields[1] if len(setting_fields) > 1 else "auto"
                method = 'MSP (LMCL)'
                l = len(feature_test)
                if threshold == "auto":
                    #prob_valid_seen = model.predict(valid_data[0])
                    #valid_unseen_idx = y_valid[~y_valid.isin(y_cols_seen)].index
                    #prob_valid_unseen = model.predict(X_valid[valid_unseen_idx])
                    seen_conf = prob_valid.max(axis=1) * -1.0
                    unseen_conf = prob_valid_ood.max(axis=1) * -1.0
                    threshold = -1.0 * estimate_best_threshold(seen_conf, unseen_conf)
                else:
                    threshold = float(threshold)

                df_seen = pd.DataFrame(prob_test, columns=le.classes_)
                df_seen['unseen'] = 0

                y_pred = df_seen.idxmax(axis=1)
                y_pred_score = df_seen.max(axis=1)
                y_pred[y_pred_score[y_pred_score < threshold].index] = 'unseen'
                cm = confusion_matrix(y_test_mask[:l], y_pred, classes)

                f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                                      title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                print(cm)
                log_pred_results(f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen,
                                 classes, pred_dir, cm, OOD_LOSS, ADV, CONT_LOSS, threshold)
