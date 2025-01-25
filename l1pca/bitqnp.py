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


def bitqnp(X_train, pcomps, sampler):
    coupler_time = 0
    sample_time = 0
    IdealSample = 0
    results_time = 0

    startQ = time.time()




    epsQ = 0
    Embeddingpath = 'Embeddings/'
    Nlimit = 175
    fconstant = (((Nlimit - 25) ** 2 - (Nlimit - 25)) / 2) + (Nlimit - 25)
    num_samples = 5  #
    num_bits = X_train.shape[1]

    B = np.zeros((num_bits, pcomps))


    rLQorg = np.zeros((X_train.shape[0], pcomps))
    K = 1
    for k in range(pcomps):
        startp = time.time()
        if k > 0:
            Xtemp = X_train - np.dot(np.dot(rLQorg[:, :(k)], rLQorg[:, :(k)].T), X_train)
        else:
            Xtemp = X_train  


        J = np.dot(-Xtemp.T, Xtemp)
        J[np.diag_indices_from(J)] *= 0.5
        scale = np.abs(J[num_bits - 1, num_bits - 1])  
        J = J / scale


        num_bits2 = num_bits

        if num_bits2 <= Nlimit:
            n, m = np.triu_indices(len(J))
            J = {(n[i], m[i]): J[n[i], m[i]] for i in range(len(n))}

            h = {}
            EndCoupler = time.time()
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
                if (len(n) + len(diagonal_indices[0])) < fconstant:
                    n = np.concatenate((diagonal_indices[0], n))
                    m = np.concatenate((diagonal_indices[1], m))
                else:
                    diagflag = 0

            J = {(n[i], m[i]): J[n[i], m[i]] for i in range(len(n))}
            h = {}
            EndCoupler = time.time()
        coupler_time = coupler_time + EndCoupler - startp
        response = sampler.sample_ising(h, J, num_reads=num_samples, label=('MultiNP'))

        IdealSample = IdealSample + (response.info['timing']['qpu_access_time'] + response.info['timing'][
            'total_post_processing_time']) * 1e-6
        EndSample = time.time()
        sample_time = sample_time + EndSample - EndCoupler




        btemp = response.record['sample'][0]  
        B[:,k]=btemp
        rtemp = np.dot(Xtemp, btemp)
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
