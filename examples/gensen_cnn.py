# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Clone GenSen repo here: https://github.com/Maluuba/gensen.git
And follow instructions for loading the model used in batcher
"""

from __future__ import absolute_import, division, unicode_literals

import re
import sys
import rcca
import logging
import numpy as np
import pandas as pd
# import GenSen package
from gensen_model import GenSen, GenSenSingle

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def kcca_transform(embed1, embed2, ndim, reg, gsigma):
    kcca = rcca.CCA(reg=reg, numCC=ndim, kernelcca=True, ktype='gaussian', gausigma=gsigma)
    cancomps = kcca.train([embed1, embed2]).comps
    return 0.5 * (cancomps[0] + cancomps[1])


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
    
    _, gensen_embeddings = params.gensen.get_representation(
        emb_df['Review'].tolist(), pool='last', return_numpy=True, tokenize=True
    )

    emb_df['CNN'] = emb_df[params.cnn_emb_type].transform(np.fromstring, sep=' ')
    # emb_df['Gensen'] = gensen_embeddings

    gensen_dim = len(gensen_embeddings[0])
    cnn_dim = 284
    cca_xdim = min(gensen_dim, cnn_dim)
    cnn_embeddings = np.asarray(emb_df['CNN'].tolist())
    cca_x = concat_emb = np.concatenate((gensen_embeddings, cnn_embeddings), axis=1)
    # cca_x = kcca_transform(gensen_embeddings, cnn_embeddings, cca_xdim, 0.01, 2.5)
    # cca_x = cca_transform(gensen_embeddings, cnn_embeddings, cca_xdim, 500)
    emb_df['CCA'] = pd.Series(map(lambda x: [x], cca_x)).apply(lambda x: x[0])

    params.embeddings = emb_df
    logging.info('Loaded Embeddings file')


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    emb_df = params.embeddings

    for sent in batch:
        sent = ' '.join(sent)
        # gensen_emb = emb_df[emb_df['Review'] == sent]['Gensen'].tolist()
        cca_emb = emb_df[emb_df['Review'] == sent]['CNN'].tolist()
        # assert len(gensen_emb) != 0, 'Gensen embeddings for "' + sent + '" not found! ' + str(len(gensen_emb))
        assert len(cca_emb) != 0, 'CCA embeddings for "' + sent + '" not found! ' + str(len(cca_emb))
        # gensen_emb = np.fromstring(bert_emb[0], sep=' ')
        # cnn_emb = np.fromstring(cnn_emb[0], sep=' ')
        # concat_emb = np.concatenate((bert_emb, cnn_emb))
        embeddings.append(cca_emb[0])

    embeddings = np.vstack(embeddings)
    return embeddings


# Load GenSen model
gensen_1 = GenSenSingle(
    model_folder='../../gensen/data/models',
    filename_prefix='nli_large_bothskip',
    pretrained_emb='../../gensen/data/embedding/glove.840B.300d.h5'
)
gensen_2 = GenSenSingle(
    model_folder='../../gensen/data/models',
    filename_prefix='nli_large_bothskip_parse',
    pretrained_emb='../../gensen/data/embedding/glove.840B.300d.h5'
)
gensen_encoder = GenSen(gensen_1, gensen_2)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['gensen'] = gensen_encoder

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
