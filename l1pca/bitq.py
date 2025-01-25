import numpy as np
from scipy import linalg
from utils import *
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
import dimod
from dimod import BinaryQuadraticModel
import dwave.system
import dwave.inspector
import minorminer
import minorminer.layout as mml
from minorminer.layout.placement import Placement, closest, intersection
import dwave_networkx as dnx
import networkx as nx
import struct
import os
from os.path import join
from dwave.samplers import SimulatedAnnealingSampler
import pickle
import time


def bitq(X_train, pcomps, sampler, epsQ):
    coupler_time = 0
    sample_time = 0
    IdealSample = 0
    results_time = 0

    startQ = time.time()

    Embeddingpath = 'Embeddings_' + str(pcomps) + '/'
    Nlimit = 175
    fconstant = (((Nlimit - 25) ** 2 - (Nlimit - 25)) / 2) + (Nlimit - 25)
    num_samples = 10  #
    num_bits = X_train.shape[1]

    Xtemp = X_train


    rLQorg = np.zeros((Xtemp.shape[0], pcomps))
    K = pcomps

    J = np.dot(-Xtemp.transpose(), Xtemp)
    J[np.diag_indices_from(J)] *= 0.5
    scale = np.abs(J[0, 0])
    J = J / scale

    if ((K * K - K) / 2 + K) * ((num_bits * num_bits - num_bits) / 2 + num_bits) < fconstant:
        n, m = np.triu_indices(len(J))
    else:
        Ntrain = len(J)
        diag_indices = np.diag_indices(Ntrain)
        n = diag_indices[0]
        m = diag_indices[1]
        offset = 0
        diagflag = 1
        while diagflag:
            offset = offset + 1

            # Get the indices of the diagonal with offset
            diagonal_indices = np.diag_indices(Ntrain - offset)

            # Modify the indices based on the offset
            diagonal_indices = (diagonal_indices[0], diagonal_indices[1] + offset)
            if ((K * K - K) / 2 + K) * (len(n) + len(diagonal_indices[0])) < fconstant:
                n = np.concatenate((diagonal_indices[0], n))
                m = np.concatenate((diagonal_indices[1], m))
            else:
                diagflag = 0


    J = {(n[i] + k1 * (num_bits), m[i] + k2 * (num_bits)): (
        (K/(epsQ))*J[n[i], m[i]] if k1 == k2 else -J[n[i], m[i]]) for i in range(len(n)) for k1 in range(K)
         for k2 in range(k1, K)}
    h = {}
    EndCoupler = time.time()
    coupler_time = coupler_time + EndCoupler - startQ




    response = sampler.sample_ising(h, J, num_reads=num_samples, label=('Multi'))

    IdealSample = IdealSample + (response.info['timing']['qpu_access_time'] + response.info['timing'][
        'total_post_processing_time']) * 1e-6
    EndSample = time.time()
    sample_time = sample_time + EndSample - EndCoupler




    B = np.zeros((num_bits,K))
    btemp = response.record['sample'][0] 
    for k in range(K):
        btempv = btemp[k * num_bits:((k + 1) * num_bits)]
        B[:,k] = btempv
        rtemp = np.dot(Xtemp, btempv)
        rLQorg[:, k] = rtemp / np.linalg.norm(rtemp, 2)
    results_time = results_time + time.time() - EndSample



    StartPost = time.perf_counter_ns()
    rLQ = procrustes(rLQorg)
    EndPost = time.perf_counter_ns()
    PostTime = (EndPost - StartPost) * 1e-9



    endQ = time.time()
    # Subtract Start Time from The End Time
    total_timeQ = endQ - startQ
    Sample_Delay = sample_time - IdealSample
    Ideal_timeQ = IdealSample + coupler_time + results_time + PostTime



    print('\nQPU')
    print('N = ' + str(num_bits) + ', K = ' + str(K))
    print("Execution Time = " + str(total_timeQ))
    print("Ideal Time = " + str(Ideal_timeQ))


    return rLQ, rLQorg, B, Ideal_timeQ, total_timeQ
