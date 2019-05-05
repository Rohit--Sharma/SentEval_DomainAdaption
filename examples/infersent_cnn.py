# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import os
import re
import io
import sys
import rcca
import torch
import logging
import numpy as np
import pandas as pd

# get models.py from InferSent repo
from models import InferSent

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = 'glove/glove.840B.300d.txt'  # or fasttext/crawl-300d-2M.vec for V2
MODEL_PATH = 'infersent1.pkl'
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def kcca_transform(embed1, embed2, ndim, reg, gsigma):
    kcca = rcca.CCA(reg=reg, numCC=ndim, kernelcca=True, ktype='gaussian', gausigma=gsigma)
    cancomps = kcca.train([embed1, embed2]).comps
    return 0.5 * (cancomps[0] + cancomps[1])


def cca_transform(embed1, embed2, ndim, n_iter):
    cca = rcca.CCA(reg=0.01, numCC=ndim, kernelcca=False)
    cancomps = cca.train([embed1, embed2]).comps
    return 0.5 * (cancomps[0] + cancomps[1])


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

    emb_file_path = params.task_path + '/downstream/' + params.current_task + '/' + params.current_task.lower() + '.tsv'
    emb_df = pd.read_csv(emb_file_path, delimiter='\t', encoding='utf-8', quotechar=u'\ua000', engine='python')
    emb_df['Review'] = emb_df['Review'].apply(lambda rev: re.sub(' +', ' ', rev).strip())
    emb_df['Review'] = emb_df['Review'].apply(lambda rev: ' '.join(rev.split()))
    emb_df['InferSent'] = emb_df['Review'].apply(lambda rev: params.infersent.encode(rev, bsize=1, tokenize=False)[0])
    emb_df['CNN'] = emb_df[params.cnn_emb_type].transform(np.fromstring, sep=' ')

    infersent_dim = len(emb_df['InferSent'][0])
    cnn_dim = 284
    cca_xdim = min(infersent_dim, cnn_dim)
    infersent_embeddings = np.asarray(emb_df['InferSent'].tolist())
    cnn_embeddings = np.asarray(emb_df['CNN'].tolist())
    # cca_x = np.concatenate((infersent_embeddings, cnn_embeddings), axis=1)
    # cca_x = kcca_transform(infersent_embeddings, cnn_embeddings, cca_xdim, 0.01, 2.5)
    cca_x = cca_transform(infersent_embeddings, cnn_embeddings, cca_xdim, 500)
    emb_df['CCA'] = pd.Series(map(lambda x: [x], cca_x)).apply(lambda x: x[0])

    params.embeddings = emb_df
    logging.info('Loaded Embeddings file')


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    emb_df = params.embeddings

    for sent in batch:
        sent = ' '.join(sent)
        cca_emb = emb_df[emb_df['Review'] == sent]['CCA'].tolist()
        assert len(cca_emb) != 0, 'CCA embeddings for "' + sent + '" not found! ' + str(len(cca_emb))
        embeddings.append(cca_emb[0])

    embeddings = np.vstack(embeddings)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    n_expts = int(sys.argv[1])
    cnn_embeddings = sys.argv[2]
    transfer_tasks = sys.argv[3:]
    acc = {}

    params_senteval['cnn_emb_type'] = cnn_embeddings

    for expt in range(n_expts):
        params_senteval['seed'] = expt
        # Load InferSent model
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.set_w2v_path(PATH_TO_W2V)

        params_senteval['infersent'] = model.cuda()

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        results = se.eval(transfer_tasks)
        print(results)
        for task in transfer_tasks:
            if task not in acc:
                acc[task] = []
            acc[task].append(results[task]['acc'])
    
    for task in transfer_tasks:
        print(task, np.mean(acc[task]), '+=', np.std(acc[task]))
