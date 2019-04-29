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


# SentEval prepare and batcher
def prepare(params, samples):
    emb_file_path = params.task_path + '/downstream/' + params.current_task + '/' + params.current_task.lower() + '.tsv'
    emb_df = pd.read_csv(emb_file_path, delimiter='\t')
    emb_df['Review'] = emb_df['Review'].apply(lambda rev: re.sub(' +', ' ', rev).strip())
    params.embeddings = emb_df
    logging.info('Loaded Embeddings file')
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    emb_df = params.embeddings

    for sent in batch:
        sent = ' '.join(sent)
        bert_emb = emb_df[emb_df['Review'] == sent]['BERT'].tolist()
        assert len(bert_emb) != 0, 'BERT embeddings for "' + sent + '" not found! ' + str(len(bert_emb))
        embeddings.append(np.fromstring(bert_emb[0], sep=' '))

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    n_expts = 10
    acc = {'Amazon': [], 'Yelp': []}
    transfer_tasks = ['Amazon', 'Yelp']
    for expt in range(n_expts):
        params_senteval['seed'] = expt

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
        #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
        #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
        #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
        #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
        #                   'OddManOut', 'CoordinationInversion']
        # transfer_tasks = ['Amazon']
        results = se.eval(transfer_tasks)
        print(results)
        for task in transfer_tasks:
            acc[task].append(results[task]['acc'])
    
    for task in transfer_tasks:
        print(task, np.mean(acc[task]), '+=', np.std(acc[task]))
