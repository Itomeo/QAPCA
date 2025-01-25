import numpy as np # linear algebra
from numpy.linalg import norm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as font_manager

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

import l1pca

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
import random
import os
from os.path import join





###########Test Parameters##########
rev = 128

num_train_samples = 20
n_components=4
runtimes = 100
newembeddings = 0 #If TRUE forces the generation of new embeddings on the annealer
corruptperr = 0.2 #percentage of outliers
path = f'rev{rev}/'
Nlimit = 175
fconstant = (((Nlimit - 25) ** 2 - (Nlimit - 25)) / 2) + (Nlimit - 25)
independentvar = [10,15,20,25,30, 35, 40]
epsQ = 100



###########Import WBCD data##########
df=pd.read_csv("breast+cancer+wisconsin+diagnostic/data.csv")
numeric_cols = df.select_dtypes(include=['number']).columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
# Standardize the features
scaler = RobustScaler()
df_numeric_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Reinsert the non-numeric columns
df = pd.concat([df_numeric_scaled, df[non_numeric_cols]], axis=1)

dftrain=df.iloc[:round(df.shape[0]/2)]
dftest=df.iloc[round(df.shape[0]/2):]

dfb=dftrain[(dftrain.diagnosis=='B')]
dfm=dftrain[(dftrain.diagnosis=='M')]




###########Begin Test##########

for run in range(runtimes):
    pcomps = n_components
    for indpnum in range(np.asarray(independentvar).shape[0]):
        ###########Load Training Data##########
        num_train_samples = independentvar[indpnum]
        num_bits = num_train_samples
        num_samples = num_train_samples
        numcorrupt = round(corruptperr * num_train_samples)
        


        Xb = dfb.drop(columns=['id', 'diagnosis'])  # Exclude the diagnosis column

        dfXb = Xb.values  # returns a numpy array
        scaler = preprocessing.StandardScaler()
        dfXb_scaled = dfXb
        Xb = pd.DataFrame(dfXb_scaled, columns=Xb.columns)

        train_sampleb = list(range(dfb.shape[0]))
        random.shuffle(train_sampleb)

        Xm = dfm.drop(columns=['id', 'diagnosis'])  # Exclude the diagnosis column
        dfXm = Xm.values  # returns a numpy array
        scaler = preprocessing.StandardScaler()
        dfXm_scaled = dfXm
        Xm = pd.DataFrame(dfXm_scaled, columns=Xm.columns)

        train_samplem = list(range(dfm.shape[0]))
        random.shuffle(train_samplem)


        Xtemp = Xb.iloc[train_sampleb[:num_train_samples]]
        Xtemp2 = Xm.iloc[train_samplem[:num_train_samples]]

        temp = Xtemp.iloc[(num_train_samples - numcorrupt):num_train_samples]
        Xtemp.iloc[(num_train_samples - numcorrupt):num_train_samples] = Xtemp2.iloc[(num_train_samples - numcorrupt):num_train_samples]
        Xtemp2.iloc[(num_train_samples - numcorrupt):num_train_samples] = temp

        Xb = Xtemp
        Xm = Xtemp2
        Xb_scaled = Xb.to_numpy()
        Xm_scaled = Xm.to_numpy()

        X = dftest.drop(columns=['id', 'diagnosis'])  # Exclude the diagnosis column
        X_scaled = X.to_numpy()#scaler.fit_transform(X)
        # Target variable (diagnosis)
        y = df['diagnosis']

        ###########L2-PCA##########
        pca = PCA(n_components=n_components,
                  svd_solver='full')  # Specify the number of components to retain--could be two or three---depends

        startL2 = time.time()
        Ql2_pcab = pca.fit_transform(Xb_scaled.T)
        endL2 = time.time()
        timeL2b = endL2 - startL2
        for k in range(n_components):
            Ql2_pcab[:, k] = Ql2_pcab[:, k] / norm(Ql2_pcab[:, k])

        startL2 = time.time()
        Ql2_pcam = pca.fit_transform(Xm_scaled.T)
        endL2 = time.time()
        timeL2m = endL2 - startL2
        print("timel2=" + str(timeL2m))
        for k in range(n_components):
            Ql2_pcam[:, k] = Ql2_pcam[:, k] / norm(Ql2_pcam[:, k])

        ###########L1-BF##########
        startBF = time.time()
        (Qlbf_pcab, Bbfb, itertemp) = l1pca.bitflipping(Xb_scaled.T, n_components, Qinit=[], tol=1e-6)
        endBF = time.time()
        timeBFb = endBF - startBF
        
        startBF = time.time()
        (Qlbf_pcam, Bbfm, itertemp) = l1pca.bitflipping(Xm_scaled.T, n_components, Qinit=[], tol=1e-6)
        endBF = time.time()
        timeBFm = endBF - startBF
        print("timeBF=" + str(timeBFm))


        ###########QAPCA-R##########
        Embeddingpathnp = 'Embeddings/'
        if newembeddings == 0:
            max_retries = 15
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if (retry_count > max_retries - 5):
                        samplernp = l1pca.embedding(Embeddingpathnp, 1, num_bits, num_samples, Nlimit, fconstant)
                    with open(Embeddingpathnp + 'embedding_' + str(num_bits) + '.dat', 'rb') as binary_file:
                        # Call load method to deserialize
                        s = binary_file.read()
                    s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
                    embeddingtnp = s_new.info['embedding_context']['embedding']
                    x = np.asarray(num_bits)
                    samplernp = FixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}), embeddingtnp)
                    break
                except Exception as e:
                    print(f"Error occurred: {e}")
                    retry_count += 1
                    print(f"Retrying... (Attempt {retry_count})")
                    time.sleep(1)  # Wait for 1 second before retrying
        else:
            samplernp = l1pca.embedding(Embeddingpathnp, 1, num_bits, num_samples, Nlimit, fconstant)



        Qlqnp_pcab, temp_org, Bqnpb, Ideal_timeQNPb, total_timeQNPb = l1pca.bitqnp(Xb_scaled.T, n_components, samplernp)
        Qlqnp_pcam, temp_org, Bqnpm, Ideal_timeQNPm, total_timeQNPm = l1pca.bitqnp(Xm_scaled.T, n_components, samplernp)
        print("timeQAPCAR=" + str(Ideal_timeQNPm))


        ###########QAPCA##########
        solver = 'bitq'
        if solver == 'bitq':
            Embeddingpath = 'Embeddings_' + str(n_components) + '/'
            num_samples = 5
            with open(Embeddingpath + 'embedding_5.dat', 'rb') as binary_file:
                # Call load method to deserialize
                s = binary_file.read()
                s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
            embedding = s_new.info['embedding_context']['embedding']
            sampler = FixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}), embedding)
            pcomps = n_components
            num_bits = num_train_samples

            if newembeddings == 0:

                max_retries = 15
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        if (retry_count > max_retries - 5):
                            sampler = l1pca.embedding(Embeddingpath, pcomps, num_bits, num_samples, Nlimit, fconstant)

                        with open(Embeddingpath + 'embedding_' + str(num_bits) + '.dat', 'rb') as binary_file:
                            # Call load method to deserialize
                            s = binary_file.read()
                        s_new = dimod.SampleSet.from_serializable(pickle.loads(s))
                        embeddingt = s_new.info['embedding_context']['embedding']
                        x = np.asarray(num_bits)
                        sampler = FixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}),
                                                          embeddingt)

                        break

                    except Exception as e:
                        print(f"Error occurred: {e}")
                        retry_count += 1
                        print(f"Retrying... (Attempt {retry_count})")
                        time.sleep(1)  # Wait for 1 second before retrying


            else:
                sampler = l1pca.embedding(Embeddingpath, pcomps, num_bits, num_samples, Nlimit, fconstant)



        print("indp=" + str(independentvar[indpnum]))
        pathsub = path + f'indp={independentvar[indpnum]}/'
        os.makedirs(pathsub, exist_ok=True)


        Qlq_pcab1, Qlq_pcab1ob, Bqb, Ideal_timeQb, total_timeQb = l1pca.bitq(Xb_scaled.T, n_components, sampler, epsQ)
        Qlq_pcam1, Qlq_pcam1ob, Bqm, Ideal_timeQm, total_timeQm = l1pca.bitq(Xm_scaled.T, n_components, sampler, epsQ)
        print("timeQAPCA=" + str(Ideal_timeQm))
        
        
        ###########Store Results##########
        timeArraytemp = np.vstack((Ideal_timeQb, total_timeQb, Ideal_timeQNPb, total_timeQNPb, timeBFb, timeL2b, Ideal_timeQm, total_timeQm, Ideal_timeQNPm, total_timeQNPm, timeBFm, timeL2m))
        np.savetxt(pathsub + 'Time' + str(rev) + '_' + str(run) + '.dat', timeArraytemp, fmt='%s', delimiter='\t',header='L1Qb,L1Qbtot,L1QNPb,L1QNPbtot,L1BFb,L1Qm,L1Qmtot,L1QNPm,L1QNPmtot,L1BFm', comments='')


        QArraytemp = np.vstack((Ql2_pcab,Qlq_pcab1,Qlbf_pcab,Qlqnp_pcab,Ql2_pcam,Qlq_pcam1,Qlbf_pcam,Qlqnp_pcam))
        np.savetxt(pathsub+'Q'+str(rev)+'_'+str(run)+'.dat', QArraytemp, fmt='%s', delimiter='\t', header='L2,L1Q,LBF,L1QNP', comments='')


        BArraytemp = np.vstack((Bqb,Bbfb,Bqnpb,Bqm,Bbfm,Bqnpm))
        np.savetxt(pathsub+'B'+str(rev)+'_'+str(run)+'.dat', BArraytemp, fmt='%s', delimiter='\t', header='L1Q,LBF,L1QNP', comments='')

        ReconErrl2b = norm(Xb_scaled.T - Ql2_pcab @ Ql2_pcab.T @ Xb_scaled.T)
        ReconErrl2m = norm(Xm_scaled.T - Ql2_pcam @ Ql2_pcam.T @ Xm_scaled.T)
        ReconErrlqb = norm(Xb_scaled.T - Qlq_pcab1 @ Qlq_pcab1.T @ Xb_scaled.T)
        ReconErrlqm = norm(Xm_scaled.T - Qlq_pcam1 @ Qlq_pcam1.T @ Xm_scaled.T)
        ReconErrlbfb = norm(Xb_scaled.T - Qlbf_pcab @ Qlbf_pcab.T @ Xb_scaled.T)
        ReconErrlbfm = norm(Xm_scaled.T - Qlbf_pcam @ Qlbf_pcam.T @ Xm_scaled.T)
        ReconErrlqnpb = norm(Xb_scaled.T - Qlqnp_pcab @ Qlqnp_pcab.T @ Xb_scaled.T)
        ReconErrlqnpm = norm(Xm_scaled.T - Qlqnp_pcam @ Qlqnp_pcam.T @ Xm_scaled.T)

        ReconErrArraytemp = np.vstack((ReconErrl2b, ReconErrlqb, ReconErrlbfb, ReconErrlqnpb, ReconErrl2m, ReconErrlqm, ReconErrlbfm, ReconErrlqnpm))
        np.savetxt(pathsub + 'ReconErr' + str(rev) + '_' + str(run) + '.dat', ReconErrArraytemp, fmt='%s', delimiter='\t',header='L2,L1Q,LBF,L1QNP', comments='')




###########Create Plots##########

timeIdealQbArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeTotalQbArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeIdealQNPbArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeTotalQNPbArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeBFbArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeL2bArray = np.zeros((np.asarray(independentvar).shape[0],1))

timeIdealQmArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeTotalQmArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeIdealQNPmArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeTotalQNPmArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeBFmArray = np.zeros((np.asarray(independentvar).shape[0],1))
timeL2mArray = np.zeros((np.asarray(independentvar).shape[0],1))


ReconErrl2bArray = np.zeros((np.asarray(independentvar).shape[0],1))
ReconErrlqbArray = np.zeros((np.asarray(independentvar).shape[0],1))
ReconErrlbfbArray = np.zeros((np.asarray(independentvar).shape[0],1))
ReconErrlqnpbArray = np.zeros((np.asarray(independentvar).shape[0],1))

ReconErrl2mArray = np.zeros((np.asarray(independentvar).shape[0],1))
ReconErrlqmArray = np.zeros((np.asarray(independentvar).shape[0],1))
ReconErrlbfmArray = np.zeros((np.asarray(independentvar).shape[0],1))
ReconErrlqnpmArray = np.zeros((np.asarray(independentvar).shape[0],1))


for run in range(runtimes):
    for indpnum in range(np.asarray(independentvar).shape[0]):
        num_train_samples = independentvar[indpnum]
        print("indp=" + str(independentvar[indpnum]))
        pathsub = path + f'indp={independentvar[indpnum]}/'
        os.makedirs(pathsub, exist_ok=True)


        timeArraytemp = np.loadtxt(pathsub + 'Time' + str(rev) + '_' + str(run) + '.dat', skiprows=1)
        timeIdealQbArray[indpnum] = timeIdealQbArray[indpnum] + timeArraytemp[0]/runtimes
        timeTotalQbArray[indpnum] = timeTotalQbArray[indpnum] + timeArraytemp[1]/runtimes
        timeIdealQNPbArray[indpnum] = timeIdealQNPbArray[indpnum] + timeArraytemp[2]/runtimes
        timeTotalQNPbArray[indpnum] = timeTotalQNPbArray[indpnum] + timeArraytemp[3]/runtimes
        timeBFbArray[indpnum] = timeBFbArray[indpnum] + timeArraytemp[4]/runtimes
        timeL2bArray[indpnum] = timeL2bArray[indpnum] + timeArraytemp[5]/runtimes

        timeIdealQmArray[indpnum] = timeIdealQmArray[indpnum] + timeArraytemp[6]/runtimes
        timeTotalQmArray[indpnum] = timeTotalQmArray[indpnum] + timeArraytemp[7]/runtimes
        timeIdealQNPmArray[indpnum] = timeIdealQNPmArray[indpnum] + timeArraytemp[8]/runtimes
        timeTotalQNPmArray[indpnum] = timeTotalQNPmArray[indpnum] + timeArraytemp[9]/runtimes
        timeBFmArray[indpnum] = timeBFmArray[indpnum] + timeArraytemp[10]/runtimes
        timeL2mArray[indpnum] = timeL2mArray[indpnum] + timeArraytemp[11]/runtimes


        ReconErrArraytemp = np.loadtxt(pathsub + 'ReconErr' + str(rev) + '_' + str(run) + '.dat', skiprows=1)
        ReconErrl2bArray[indpnum] = ReconErrl2bArray[indpnum] + ReconErrArraytemp[0]/runtimes
        ReconErrlqbArray[indpnum] = ReconErrlqbArray[indpnum] + ReconErrArraytemp[1]/runtimes
        ReconErrlbfbArray[indpnum] = ReconErrlbfbArray[indpnum] + ReconErrArraytemp[2]/runtimes
        ReconErrlqnpbArray[indpnum] = ReconErrlqnpbArray[indpnum] + ReconErrArraytemp[3]/runtimes
        
        ReconErrl2mArray[indpnum] = ReconErrl2mArray[indpnum] + ReconErrArraytemp[4]/runtimes
        ReconErrlqmArray[indpnum] = ReconErrlqmArray[indpnum] + ReconErrArraytemp[5]/runtimes
        ReconErrlbfmArray[indpnum] = ReconErrlbfmArray[indpnum] + ReconErrArraytemp[6]/runtimes
        ReconErrlqnpmArray[indpnum] = ReconErrlqnpmArray[indpnum] + ReconErrArraytemp[7]/runtimes



tnrfont = {'fontname':'Times New Roman'}
tnrfont2 = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=16)
plt.rcParams["mathtext.fontset"] = "cm"

arr1,=plt.plot(independentvar, timeL2bArray+timeL2mArray, 'bo-', linewidth=2, markersize=12, label = 'L2')
arr2,=plt.plot(independentvar, timeIdealQbArray+timeIdealQmArray, 'r+-', linewidth=2, markersize=12, label = 'QAPCA')
arr3,=plt.plot(independentvar, timeBFbArray+timeBFmArray, 'kd-', linewidth=2, markersize=12, label = 'L1 BF')
arr4,=plt.plot(independentvar, timeIdealQNPbArray+timeIdealQNPmArray, 'gs-', linewidth=2, markersize=12, label = 'QAPCA NP')
plt.ylabel("Time", fontname="Times New Roman", fontsize=16)
plt.xlabel("Samples", fontname="Times New Roman", fontsize=16)
plt.legend([arr2, arr4, arr1, arr3], ['QAPCA (Ours)', 'QAPCA-R (Ours)', 'SVD [17]', 'L1-BF [4]'],prop=tnrfont2)#, 'QAPCA NP'
matplotlib.pyplot.xticks(independentvar)
plt.show()


plt.show(block=True)
input()


arr1,=plt.plot(independentvar, ReconErrl2bArray+ReconErrl2mArray, 'bo-', linewidth=2, markersize=12, label = 'L2')
arr2,=plt.plot(independentvar, ReconErrlqbArray+ReconErrlqmArray, 'r+-', linewidth=2, markersize=12, label = 'QAPCA')
arr3,=plt.plot(independentvar, ReconErrlbfbArray+ReconErrlbfmArray, 'kd-', linewidth=2, markersize=12, label = 'L1 BF')

arr4,=plt.plot(independentvar, ReconErrlqnpbArray+ReconErrlqnpmArray, 'gs-', linewidth=2, markersize=12, label = 'QAPCA NP')
plt.ylabel("Reconstruction Error", fontname="Times New Roman", fontsize=12)
plt.xlabel("Samples", fontname="Times New Roman", fontsize=12)
plt.legend([arr2, arr4, arr1, arr3],  ['QAPCA (Ours)', 'QAPCA-R (Ours)', 'SVD [17]', 'L1-BF [4]'],prop=tnrfont2)#, arr4, arr5, 'QAPCA NP','QAPCA OB'
matplotlib.pyplot.xticks(independentvar)
plt.show()


plt.show(block=True)
input()



































