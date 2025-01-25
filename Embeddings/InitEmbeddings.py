import numpy as np
import math
import imageio as iio
from PIL import Image
from IPython.display import Image as IPy
from pathlib import Path
from numpy.linalg import norm 
import datetime
import dimod
import pickle
from utils import * # It contains functions for threat the data (I/O, encoding/decoding) and metrics for evaluations 
# import the builtin time module
import time
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
from dimod import BinaryQuadraticModel 
import numpy.lib.recfunctions as rfn
import warnings
import concurrent.futures
import dwave.system
import dwave.inspector
import random
from sklearn.decomposition import PCA
from compcand import *
import minorminer
import minorminer.layout as mml
#import minorminer.layout.placement
#from minorminer.placement import Placement, closest, intersection
from minorminer.layout.placement import Placement, closest, intersection
import dwave_networkx as dnx
import networkx as nx
from scipy.stats import ortho_group
from scipy.stats import multivariate_normal
warnings.simplefilter("ignore", DeprecationWarning)

num_samples =1

StartPre = time.time()

print('pegasus_graph')

sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}), )

for num_bits in [150]:
    tempJ = np.ones((num_bits, num_bits))
    qubo_nodes = np.asarray([[n, n, tempJ[n, n]] for n in range(len(tempJ))])
    qubo_couplers = np.asarray([[n, m, tempJ[n, m]] for n in range(len(tempJ)) for m in range(n + 1,len(tempJ))])
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:, 2]))]
    qubo_nodes = np.array([[i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)] for i in np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0,1]].max() + 1)])

    J = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers))}
    h = {}

    response = sampler.sample_ising(h, J, num_reads=num_samples, label=('Init Embeddings'))
    s = pickle.dumps(response.to_serializable())
    file = 'embedding_'+str(num_bits)+'.dat'
    with open(file, "wb") as binary_file:
        # Write bytes to file
        binary_file.write(s)


