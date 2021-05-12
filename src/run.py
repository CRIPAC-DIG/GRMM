import torch
import numpy as np
import os
import sys
from time import time
import random
import warnings 
warnings.filterwarnings("ignore") 
from utility.parser import parse_args
args = parse_args()

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

from utility.batch_test import data_generator, pad_sequences, words_lookup, test
from utility.models import *


if __name__ == '__main__':
    device = torch.device("cuda:%d"%args.gpu_id)
    users_to_test = list(data_generator.test_set.keys())
    config = dict()
    config['n_docs'] = data_generator.n_docs
    config['n_qrls'] = data_generator.n_qrls
    config['n_words'] = data_generator.n_words
    if args.model == 'GRMM':
        config['docs_adj'] = np.load("../Data/{}/doc_adj.npy".format(args.dataset))
    config['idf_dict'] = np.load("../Data/{}/idf.npy".format(args.dataset), allow_pickle=True).item()

    model = eval(args.model)(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2)
    precision = []
    ndcg = []
    best_ret = 0
    best_epoch = 0
    for epoch in range(args.epoch):
        t0 = time()
        n_batch = 32
        total_loss = 0
        model.train()
        for idx in range(n_batch):
            qrls, pos_docs, neg_docs = data_generator.sample()
            pos_docs_words = pad_sequences(words_lookup(pos_docs), maxlen=args.doc_len, value=config['n_words'])
            neg_docs_words = pad_sequences(words_lookup(neg_docs), maxlen=args.doc_len, value=config['n_words'])
            l = [data_generator.qrl_word_list[i] for i in qrls]
            qrls_words = pad_sequences(l, maxlen=args.qrl_len, value=config['n_words'])
            qrls_words = torch.tensor(qrls_words).long().to(device)
            pos_docs_words = torch.tensor(pos_docs_words).long().to(device)
            neg_docs_words = torch.tensor(neg_docs_words).long().to(device)

            pos_scores = model(qrls_words, pos_docs_words, pos_docs)
            neg_scores = model(qrls_words, neg_docs_words, neg_docs)
            loss = torch.max(torch.zeros_like(pos_scores).float().to(device), (1 - pos_scores + neg_scores))
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.cpu().detach().numpy()
            optimizer.step()
        ret = test(model, users_to_test)
        precision.append(ret[0])
        ndcg.append(ret[1])
        if args.verbose:
            print("epoch:%d"%epoch, "loss:%.4f"%(total_loss/args.batch_size)
            , "p@20:%.3f"%precision[-1], "ndcg@20:%.3f"%ndcg[-1])
        if ret[1] > best_ret:
            best_ret = ret[1]
            best_epoch = epoch
            # model.save('{}.{}.f{}.best.model'.format(args.model, args.dataset, str(args.fold)))

    print("P@20:", precision[best_epoch])
    print("ndcg@20:", ndcg[best_epoch])
