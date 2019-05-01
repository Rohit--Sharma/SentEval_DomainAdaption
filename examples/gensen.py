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

import sys
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

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    _, reps_h_t = params.gensen.get_representation(
        batch, pool='last', return_numpy=True, tokenize=True
    )
    embeddings = reps_h_t
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
