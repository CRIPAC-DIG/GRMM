import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utility.parser import parse_args
from utility.layers import *
args = parse_args()
device = torch.device("cuda:%d"%args.gpu_id)

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_docs = config['n_docs']
        self.n_qrls = config['n_qrls']
        self.n_words = config['n_words']
        pretrained_embeddings = np.load("../Data/{}/word_embedding_300d.npy".format(args.dataset))
        l2_norm = np.sqrt((pretrained_embeddings * pretrained_embeddings).sum(axis=1))
        pretrained_embeddings = pretrained_embeddings / l2_norm[:, np.newaxis]
        pretrained_embeddings = np.concatenate([pretrained_embeddings, np.zeros([1, pretrained_embeddings.shape[-1]])], 0)
        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings),freeze=True,padding_idx=self.n_words).float()
        self._init_weights()
    def _init_weights(self):
        raise NotImplementedError 
    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)
    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
    def create_simmat(self, a_emb, b_emb):
        BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
        a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9 # avoid 0div
        b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9 # avoid 0div
        perm = b_emb.permute(0, 2, 1)
        sim = a_emb.bmm(perm)
        sim = sim / (a_denom * b_denom)
        return sim


class GRMM(BaseModel):
    def _init_weights(self):
        self.docs_adj = self.config['docs_adj']
        self.idf_dict = self.config['idf_dict']
        self.linear1 = nn.Linear(args.topk, 64) 
        self.linear2 = nn.Linear(64, 32) 
        self.linear3 = nn.Linear(32, 1) 
        self.linearz0 = nn.Linear(args.qrl_len, args.qrl_len) 
        self.linearz1 = nn.Linear(args.qrl_len, args.qrl_len) 
        self.linearr0 = nn.Linear(args.qrl_len, args.qrl_len) 
        self.linearr1 = nn.Linear(args.qrl_len, args.qrl_len) 
        self.linearh0 = nn.Linear(args.qrl_len, args.qrl_len) 
        self.linearh1 = nn.Linear(args.qrl_len, args.qrl_len) 
        self.gated = nn.Linear(1, 1) 
        self.dropout = nn.Dropout(args.dp)
    def forward(self, qrls_words, doc_words, doc_ids, test=False):
        self.idf = torch.FloatTensor([[self.idf_dict[word.item()] for word in query] for query in qrls_words]) .unsqueeze(-1).to(device)
        qrl_word_embedding = self.word_embedding(qrls_words)
        doc_word_embedding = self.word_embedding(doc_words)
        feat = self.create_simmat(qrl_word_embedding, doc_word_embedding).permute(0, 2, 1) #batch, len_d, len_q
        adj = self.docs_adj[doc_ids]
        adj = torch.FloatTensor(adj).to(device)
        for k in range(args.layers):
            x = feat
            a = adj.matmul(x)

            z0 = self.linearz0(a)
            z1 = self.linearz1(x)
            z = F.sigmoid(z0 + z1)

            r0 = self.linearr0(a)
            r1 = self.linearr1(x)
            r = F.sigmoid(r0 + r1)

            h0 = self.linearh0(a)
            h1 = self.linearh1(r*x)
            h = F.relu(h0 + h1)

            feat = h*z + x*(1-z)
            x = self.dropout(feat)


        feat = feat.permute(0, 2, 1)
        topk, _ = feat.topk(args.topk, -1) # batch, qrl, doc
        rel = F.relu(self.linear1(topk))
        rel = F.relu(self.linear2(rel))
        rel = self.linear3(rel)
        if args.idf:
            gated_weight = F.softmax(self.gated(self.idf), dim=1)
            rel = rel * gated_weight
        scores = rel.squeeze(-1).sum(-1, keepdim=True)
        if test:
            scores = scores.reshape((1, -1))
        return scores

# https://github.com/Georgetown-IR-Lab/cedr
class DRMM(BaseModel):
    def _init_weights(self):
        NBINS = 11
        HIDDEN = 5
        self.histogram = DRMMLogCountHistogram(NBINS, self.n_docs)
        self.hidden_1 = nn.Linear(NBINS, HIDDEN)
        self.hidden_2 = nn.Linear(HIDDEN, 1)
        self.idf_dict = self.config['idf_dict']
        self.gated = nn.Linear(1, 1)
    def forward(self, qrls_words, doc_words, doc_ids, test=False):
        self.idf = torch.FloatTensor([[self.idf_dict[word.item()] for word in query] for query in qrls_words]) .unsqueeze(-1).to(device)
        qrl_word_embedding = self.word_embedding(qrls_words)
        doc_word_embedding = self.word_embedding(doc_words)
        simmat = self.create_simmat(qrl_word_embedding, doc_word_embedding)
        histogram = self.histogram(simmat, doc_words, qrls_words)
        BATCH, QLEN, BINS = histogram.shape
        # histogram = histogram.permute(0, 2, 3, 1)
        output = histogram.reshape(BATCH * QLEN, BINS)
        rel = self.hidden_2(torch.relu(self.hidden_1(output))).reshape(BATCH, QLEN)
        rel = rel.unsqueeze(-1)
        gated_weight = F.softmax(self.gated(self.idf), dim=1)
        rel = rel * gated_weight
        scores = rel.squeeze(-1).sum(-1, keepdim=True)
        if test:
            scores = scores.reshape((1, -1))
        return scores

class PACRR(BaseModel):
    def _init_weights(self):
        KMAX = 2
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.ngrams = nn.ModuleList()
        for ng in range(MINGRAM, MAXGRAM+1):
            ng = PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=1)
            self.ngrams.append(ng)
        self.linear1 = nn.Linear(((MAXGRAM - MINGRAM + 1) * KMAX + 1) * args.qrl_len, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 1)
        self.idf_dict = self.config['idf_dict']
    def forward(self, qrls_words, doc_words, doc_ids, test=False):
        self.idf = torch.FloatTensor([[self.idf_dict[word.item()] for word in query] for query in qrls_words]) .unsqueeze(-1).to(device)
        qrl_word_embedding = self.word_embedding(qrls_words)
        doc_word_embedding = self.word_embedding(doc_words)
        simmat = self.create_simmat(qrl_word_embedding, doc_word_embedding)
        simmat = simmat.unsqueeze(dim=1)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)
        scores = torch.cat([scores, self.idf], dim=-1)
        scores = scores.reshape(scores.shape[0], -1)
        rel = F.relu(self.linear1(scores))
        rel = F.relu(self.linear2(rel))
        scores = self.linear3(rel)
        if test:
            scores = scores.reshape((1, -1))
        return scores

class COPACRR(BaseModel):
    def _init_weights(self):
        KMAX = 2
        NFILTERS = 32
        MINGRAM = 1
        MAXGRAM = 3
        self.TOP = 6
        self.w = 4
        self.ngrams = nn.ModuleList()
        for ng in range(MINGRAM, MAXGRAM+1):
            ng = PACRRConvMax2dModule(ng, NFILTERS, k=KMAX, channels=1)
            self.ngrams.append(ng)
        self.linear1 = nn.Linear(((MAXGRAM - MINGRAM + 1) * KMAX + 1 + self.TOP) * args.qrl_len, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 1)
        self.idf_dict = self.config['idf_dict']
    def forward(self, qrls_words, doc_words, doc_ids, test=False):
        self.idf = torch.FloatTensor([[self.idf_dict[word.item()] for word in query] for query in qrls_words]) .unsqueeze(-1).to(device)
        qrl_word_embedding = self.word_embedding(qrls_words)
        doc_word_embedding = self.word_embedding(doc_words)
        context = torch.zeros_like(doc_word_embedding).to(device)
        w = self.w
        for i in range(context.shape[1]):
            s = max(0, i-w)
            e = min(context.shape[1], i+w)
            context[:,i,:] = doc_word_embedding[:,s:e].sum(1)
        context = context / (2 * w + 1)
        querysim = self.create_simmat(qrl_word_embedding, context)
        querysim,_ = querysim.topk(self.TOP,-1)

        simmat = self.create_simmat(qrl_word_embedding, doc_word_embedding)
        simmat = simmat.unsqueeze(dim=1)
        scores = [ng(simmat) for ng in self.ngrams]
        scores = torch.cat(scores, dim=2)

        scores = torch.cat([scores, querysim ,self.idf], dim=-1)
        scores = scores.reshape(scores.shape[0], -1)
        scores = F.relu(self.linear1(scores))
        scores = F.relu(self.linear2(scores))
        scores = self.linear3(scores)
        if test:
            scores = scores.reshape((1, -1))
        return scores

class KNRM(BaseModel):
    def _init_weights(self):
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.kernels = KNRMRbfKernelBank(MUS, SIGMAS)
        self.combine = torch.nn.Linear(self.kernels.count(), 1)

    def forward(self, qrls_words, doc_words, doc_ids, test=False):
        qrl_word_embedding = self.word_embedding(qrls_words)
        doc_word_embedding = self.word_embedding(doc_words)
        simmat = self.create_simmat(qrl_word_embedding, doc_word_embedding)

        kernels = self.kernels(simmat)
        BATCH, KERNELS, QLEN, DLEN = kernels.shape
        VIEWS = 1

        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN) \
                       .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN) \
                       .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        result = kernels.sum(dim=3) # sum over document
        mask = (simmat.sum(dim=3) != 0.) # which query terms are not padding?
        result = torch.where(mask, (result + 1e-6).log(), mask.float())
        result = result.sum(dim=2) # sum over query terms
        scores = self.combine(result) # linear combination over kernels
        if test:
            scores = scores.reshape((1, -1))
        return scores

# https://github.com/NTMC-Community/MatchZoo-py
class MP(BaseModel):
    def _init_weights(self):
        self.embedding_output_dim = 300
        self.kernel_count = [16, 32]
        self.kernel_size = [[3, 3], [3, 3]]
        self.dpool_size = [3, 10]
        self.dropout_rate = 0.0
        activation = nn.ReLU()
        
        in_channel_2d = [1,*self.kernel_count[:-1]]
        conv2d = [
            self._make_conv_pool_block(ic, oc, ks, activation)
            for ic, oc, ks, in zip(in_channel_2d,
                                   self.kernel_count,
                                   self.kernel_size)
        ]
        self.conv2d = nn.Sequential(*conv2d)

        # Dynamic Pooling
        self.dpool_layer = nn.AdaptiveAvgPool2d(self.dpool_size)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        left_length = self.dpool_size[0]
        right_length = self.dpool_size[1]

        # Build output
        self.out = nn.Linear(left_length * right_length * self.kernel_count[-1], 1)

    def forward(self, qrls_words, doc_words, doc_ids, test=False):
        embed_left = self.word_embedding(qrls_words)
        embed_right = self.word_embedding(doc_words)
        embed_cross =  torch.einsum('bld,brd->blr', embed_left, embed_right).unsqueeze(1)

        # Convolution
        # shape = [B, F, L, R]
        conv = self.conv2d(embed_cross)

        # Dynamic Pooling
        # shape = [B, F, P1, P2]
        embed_pool = self.dpool_layer(conv)

        # shape = [B, F * P1 * P2]
        embed_flat = self.dropout(torch.flatten(embed_pool, start_dim=1))

        # shape = [B, *]
        scores = torch.relu(self.out(embed_flat))
        if test:
            scores = scores.reshape((1, -1))
        return scores

    @classmethod
    def _make_conv_pool_block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        activation: nn.Module
    ) -> nn.Module:
        """Make conv pool block."""
        return nn.Sequential(nn.ConstantPad2d((0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size),
            activation
        )