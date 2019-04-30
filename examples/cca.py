# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import re
import rcca
import numpy as np
import pandas as pd
import logging


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_VEC = 'glove/glove.840B.300d.txt'
# PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def cca_transform(embed1, embed2, ndim, n_iter):
    cca = rcca.CCA(reg=0.01, numCC=ndim, kernelcca=False)
    cancomps = cca.train([embed1, embed2]).comps
    return 0.5 * (cancomps[0] + cancomps[1])


# SentEval prepare and batcher
def prepare(params, samples):
    emb_file_path = params.task_path + '/downstream/' + params.current_task + '/' + params.current_task.lower() + '.tsv'
    emb_df = pd.read_csv(emb_file_path, delimiter='\t', encoding='utf-8', quotechar=u'\ua000', engine='python')
    emb_df['Review'] = emb_df['Review'].apply(lambda rev: re.sub(' +', ' ', rev).strip())
    emb_df['Review'] = emb_df['Review'].apply(lambda rev: ' '.join(rev.split()))
    emb_df['BERT'] = emb_df['BERT'].transform(np.fromstring, sep=' ')
    emb_df['CNN'] = emb_df[params.cnn_emb_type].transform(np.fromstring, sep=' ')

    bert_dim = 768
    cnn_dim = 384
    cca_xdim = min(bert_dim, cnn_dim)
    bert_embeddings = np.asarray(emb_df['BERT'].tolist())
    cnn_embeddings = np.asarray(emb_df['CNN'].tolist())
    cca_x = cca_transform(bert_embeddings, cnn_embeddings, cca_xdim, 500)
    emb_df['CCA'] = pd.Series(map(lambda x: [x], cca_x)).apply(lambda x: x[0])

    params.embeddings = emb_df
    logging.info('Loaded Embeddings file')
    return

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


# Set params for SentEval
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
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        results = se.eval(transfer_tasks)
        print(results)
        for task in transfer_tasks:
            if task not in acc:
                acc[task] = []
            acc[task].append(results[task]['acc'])
    
    for task in transfer_tasks:
        print(task, np.mean(acc[task]), '+=', np.std(acc[task]))
