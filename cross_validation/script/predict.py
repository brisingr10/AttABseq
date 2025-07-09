#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
from model import Encoder, Decoder, Predictor, SelfAttention, PositionwiseFeedforward, DecoderLayer, pack

class feature(object):
    def __init__(self,seq, dataset_name):
        self.seq = seq
        self.dataset_name = dataset_name

    def seq2onehot(self):
        aas = {'X':0,'A':1,'R':2,'N':3,'D':4,'C':5,
               'Q':6,'E':7,'G':8,'H':9,'I':10,
               'L':11,'K':12,'M':13,'F':14,'P':15,
               'S':16,'T':17,'W':18,'Y':19,'V':20}
        seq_onehot = np.zeros((len(self.seq),len(aas)))
        for i, aa in enumerate(self.seq[:]):
            if aa not in aas:
                aa = 'X'
            seq_onehot[i, (aas[aa])] = 1
        seq_onehot = seq_onehot[:,1:]
        return seq_onehot

    def seq2pssm(self):
        fasta_file = f'{self.dataset_name}aa.fasta'
        output_file = f'{self.dataset_name}.txt'
        pssm_file = f'{self.dataset_name}aa.pssm'
        with open(fasta_file, 'w') as f:
            f.write('>name\n')
            f.write(self.seq)
        f.close()
        # It is assumed that the blast tool is in the parent directory of the script directory.
        os.system(f'../ncbi-blast-2.12.0+/bin/psiblast -query {fasta_file} -db ../ncbi-blast-2.12.0+/bin/swissprot -num_iterations 3 -out {output_file} -out_ascii_pssm {pssm_file}')
        with open(pssm_file, 'r') as inputpssm:
            count = 0
            pssm_matrix = []
            for eachline in inputpssm:
                count += 1
                if count <= 3:
                    continue
                if not len(eachline.strip()):
                    break
                col = eachline.strip()
                col = col.split(' ')
                col = [x for x in col if x != '']
                col = col[2:22]
                col = [int(x) for x in col]
                oneline = col
                pssm_matrix.append(oneline)
            seq_pssm = np.array(pssm_matrix)
        return seq_pssm

def all_feature(ls, dataset_name):
    all = []
    for s in ls:
        if s==s:
            f1 = feature(s, dataset_name).seq2onehot()
            f2 = feature(s, dataset_name).seq2pssm()
            f = np.concatenate((f1,f2),axis=1)
            all.append(f)
        else:
            all.append(s)
    return all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict with AttABseq model.')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file.', required=True)
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset CSV file.', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save the prediction results.', required=True)
    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    output_path = args.output_path
    dataset_name = os.path.basename(dataset_path).split('.')[0]

    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    csv = pd.read_csv(dataset_path)
    
    if 'antibody_light_seq' in csv.columns:
        abls = csv['antibody_light_seq'].tolist()
        abhs = csv['antibody_heavy_seq'].tolist()
        agas = csv['antigen_a_seq'].tolist()
        agbs = csv['antigen_b_seq'].tolist()
        abls_m = csv['antibody_light_seq_mut'].tolist()
        abhs_m = csv['antibody_heavy_seq_mut'].tolist()
        agas_m = csv['antigen_a_seq_mut'].tolist()
        agbs_m = csv['antigen_b_seq_mut'].tolist()
        labels = csv['ddG'].tolist()

        antibodies_l = all_feature(abls, dataset_name)
        antibodies_h = all_feature(abhs, dataset_name)
        antigens_a = all_feature(agas, dataset_name)
        antigens_b = all_feature(agbs, dataset_name)
        antibodies_l_mut = all_feature(abls_m, dataset_name)
        antibodies_h_mut = all_feature(abhs_m, dataset_name)
        antigens_a_mut = all_feature(agas_m, dataset_name)
        antigens_b_mut = all_feature(agbs_m, dataset_name)

        antibodies = []
        antigens = []
        antibodies_mut = []
        antigens_mut = []
        for i in range(len(antibodies_l)):
            if isinstance(antibodies_h[i],float):
                antibodies.append(antibodies_l[i])
            else:
                antibodies.append(np.concatenate((antibodies_l[i],antibodies_h[i]),axis=0))
            if isinstance(antigens_b[i],float):
                antigens.append(antigens_a[i])
            else:
                antigens.append(np.concatenate((antigens_a[i],antigens_b[i]),axis=0))
            if isinstance(antibodies_h_mut[i],float):
                antibodies_mut.append(antibodies_l_mut[i])
            else:
                antibodies_mut.append(np.concatenate((antibodies_l_mut[i],antibodies_h_mut[i]),axis=0))
            if isinstance(antigens_b_mut[i],float):
                antigens_mut.append(antigens_a_mut[i])
            else:
                antigens_mut.append(np.concatenate((antigens_a_mut[i],antigens_b_mut[i]),axis=0))
    else:
        ab = csv['a'].tolist()
        ag = csv['b'].tolist()
        ab_m = csv['a_mut'].tolist()
        ag_m = csv['b_mut'].tolist()
        labels = csv['ddG'].tolist()

        antibodies = all_feature(ab, dataset_name)
        antigens = all_feature(ag, dataset_name)
        antibodies_mut = all_feature(ab_m, dataset_name)
        antigens_mut = all_feature(ag_m, dataset_name)

    interactions = np.array(labels)

    dataset = list(zip(antigens, antibodies, antigens_mut, antibodies_mut, interactions))

    antibody_dim = 40
    antigen_dim = 40
    hid_dim = 256
    n_layers = 3
    n_heads = 8
    pf_dim = 64
    dropout = 0.1
    kernel_size = 3

    encoder = Encoder(antibody_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(antigen_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    
    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for data in dataset:
            ag_s, ab_s, ag_m_s, ab_m_s, correct_interaction = [data]
            data_pack = pack([ag_s], [ab_s], [ag_m_s], [ab_m_s], [correct_interaction], device)
            predicted_interaction, _, _, _, _ = model.forward(*data_pack[:-1])
            predictions.append(predicted_interaction.item())

    results = pd.DataFrame({'true_ddG': interactions, 'predicted_ddG': predictions})
    results.to_csv(output_path, index=False)

    print(f'Predictions saved to {output_path}')