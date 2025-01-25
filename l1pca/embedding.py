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


def embedding(Embeddingpath, K, num_bits, num_samples, Nlimit, fconstant):
    print('Gen Samples')
    newembeddings = 1

    with open(Embeddingpath + 'embedding_5.dat', 'rb') as binary_file:
        # Call load method to deserialize
        s = binary_file.read()
        s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
    embedding = s_new.info['embedding_context']['embedding']
    x = np.asarray(num_bits)
    sampler = FixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}), embedding)

    # for num_bits in train_samples:
    print(f'num_bits = {num_bits}')
    if ((K * K - K) / 2 + K) * ((num_bits * num_bits - num_bits) / 2 + num_bits) <= fconstant:
        if newembeddings == 0:
            with open(Embeddingpath + 'embedding_' + str(num_bits) + '.dat', 'rb') as binary_file:
                # Call load method to deserialize
                s = binary_file.read()
            s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
            embedding = s_new.info['embedding_context']['embedding']

        # Maximum number of retries
        max_retries = 5

        # Retry loop
        retry_count = 0
        while retry_count < max_retries:
            try:
                if (retry_count > max_retries - 3) or (newembeddings == 1):
                    sampler2 = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}), )
                    tempJ = np.ones((num_bits, num_bits))

                    n, m = np.triu_indices(len(tempJ))
                    J = {(n[i] + k1 * (num_bits), m[i] + k2 * (num_bits)): tempJ[n[i], m[i]] for i in range(len(n))
                         for k1 in range(K) for k2 in range(k1, K)}
                    h = {}

                    response = sampler2.sample_ising(h, J, num_reads=num_samples)
                    s = pickle.dumps(response.to_serializable())
                    file = 'embedding_' + str(num_bits) + '.dat'
                    with open(Embeddingpath + file, "wb") as binary_file:
                        # Write bytes to file
                        binary_file.write(s)
                    with open(Embeddingpath + 'embedding_' + str(num_bits) + '.dat',
                              'rb') as binary_file:
                        # Call load method to deserialize
                        s = binary_file.read()
                    s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
                    embedding = s_new.info['embedding_context']['embedding']

                # Your code that may raise an error
                # For example, let's simulate a division by zero error
                sampler = FixedEmbeddingComposite(
                    DWaveSampler(solver={'topology__type': 'pegasus'}), embedding)

                # If no error is raised, break out of the loop
                break

            except Exception as e:
                print(f"Error occurred: {e}")
                retry_count += 1
                print(f"Retrying... (Attempt {retry_count})")
                time.sleep(1)  # Wait for 1 second before retrying

        if retry_count == max_retries:
            print("Max retries reached. Exiting...")
            exit()
    else:
        max_retries = 5
        if newembeddings == 0:
            with open(Embeddingpath + 'embedding_' + str(num_bits) + '.dat', 'rb') as binary_file:
                # Call load method to deserialize
                s = binary_file.read()
            s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
            embedding = s_new.info['embedding_context']['embedding']
            max_retries = 30
        # Maximum number of retries

        # Retry loop
        retry_count = 0
        while retry_count < max_retries:
            try:
                if retry_count > max_retries - 3:

                    sampler2 = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}), )
                    tempJ = np.ones((num_bits, num_bits))

                    # Define the offset
                    Ntrain = len(tempJ)
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

                    J = {(n[i] + k1 * (num_bits), m[i] + k2 * (num_bits)): tempJ[n[i], m[i]] for i in range(len(n))
                         for k1 in range(K) for k2 in range(k1, K)}
                    h = {}

                    response = sampler2.sample_ising(h, J, num_reads=num_samples)  # ,
                    s = pickle.dumps(response.to_serializable())
                    file = 'embedding_' + str(num_bits) + '.dat'
                    with open(Embeddingpath + file, "wb") as binary_file:
                        # Write bytes to file
                        binary_file.write(s)
                    with open(Embeddingpath + 'embedding_' + str(num_bits) + '.dat',
                              'rb') as binary_file:
                        # Call load method to deserialize
                        s = binary_file.read()
                    s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
                    embedding = s_new.info['embedding_context']['embedding']
                # Your code that may raise an error
                # For example, let's simulate a division by zero error
                sampler = FixedEmbeddingComposite(
                    DWaveSampler(solver={'topology__type': 'pegasus'}),
                    embedding)

                # If no error is raised, break out of the loop
                break

            except Exception as e:
                print(f"Error occurred: {e}")
                retry_count += 1
                print(f"Retrying... (Attempt {retry_count})")
                time.sleep(1)  # Wait for 1 second before retrying

        if retry_count == max_retries:
            print("Max retries reached. Exiting...")
            exit()

    print('embedding loaded')

    return sampler


