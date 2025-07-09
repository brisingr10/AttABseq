#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
import time
from model import *
import timeit
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from pytorchtools import EarlyStopping

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
    parser = argparse.ArgumentParser(description='Train AttABseq model.')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset CSV file.', required=True)
    parser.add_argument('--n_splits', type=int, default=10, help='Number of cross-validation splits.')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    n_splits = args.n_splits
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

    print('Training...')

    kf = KFold(n_splits=n_splits, shuffle=True)

    output_dir = f'../output{dataset_name.replace("AB", "").replace("S", "")}'
    analysis_dir = f'../../split/{dataset_name.replace("AB", "").replace("S", "")}analysis'
    os.makedirs(f'{output_dir}/loss_min_result', exist_ok=True)
    os.makedirs(f'{output_dir}/loss_min_model', exist_ok=True)
    os.makedirs(f'{output_dir}/best_pcc_result', exist_ok=True)
    os.makedirs(f'{output_dir}/best_pcc_model', exist_ok=True)
    os.makedirs(f'{output_dir}/best_r2_result', exist_ok=True)
    os.makedirs(f'{output_dir}/best_r2_model', exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    i=0
    for train_index, val_index in kf.split(interactions):
        antibody_dim = 40
        antigen_dim = 40
        hid_dim = 256
        n_layers = 3
        n_heads = 8
        pf_dim = 64
        dropout = 0.1
        batch = 8
        lr = 0.00001
        weight_decay = 1e-4
        iteration = 150
        kernel_size = 3
        minloss = 1000
        best_pearson = -1000
        best_r2 = -1000
        
        encoder = Encoder(antibody_dim, hid_dim, n_layers, kernel_size, dropout, device)
        decoder = Decoder(antigen_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
        
        model = Predictor(encoder, decoder, device)
        model.to(device)
        
        trainer = Trainer(model, lr, weight_decay, batch)
        tester = Tester(model)
        
        i+=1
        print(f'*************************** start training on Fold {i} ***************************')

        os.makedirs(f'{analysis_dir}/fold{i}', exist_ok=True)
        with open(f'{analysis_dir}/fold{i}/fold_indices.txt','w') as file_fold:
            file_fold.write(str(train_index))
            file_fold.write('\n')
            file_fold.write(str(val_index))
            file_fold.write('\n')
        file_fold.close()
        
        antigens_train, antigens_val = np.array(antigens)[train_index], np.array(antigens)[val_index]
        antibodies_train, antibodies_val = np.array(antibodies)[train_index], np.array(antibodies)[val_index]
        antigens_mut_train, antigens_mut_val = np.array(antigens_mut)[train_index], np.array(antigens_mut)[val_index]
        antibodies_mut_train, antibodies_mut_val = np.array(antibodies_mut)[train_index], np.array(antibodies_mut)[val_index]
        interactions_train, interactions_val = np.array(interactions)[train_index], np.array(interactions)[val_index]

        dataset_train = list(zip(antigens_train, antibodies_train, antigens_mut_train, antibodies_mut_train, interactions_train))
        dataset_val = list(zip(antigens_val, antibodies_val, antigens_mut_val, antibodies_mut_val, interactions_val))

        file_loss_min_PCCS = f'{output_dir}/loss_min_result/RECORD_{i}.txt'
        file_loss_min_model = f'{output_dir}/loss_min_model/model_{i}'
        file_best_pcc_PCCS = f'{output_dir}/best_pcc_result/RECORD_{i}.txt'
        file_best_pcc_model = f'{output_dir}/best_pcc_model/model_{i}'
        file_best_r2_PCCS = f'{output_dir}/best_r2_result/RECORD_{i}.txt'
        file_best_r2_model = f'{output_dir}/best_r2_model/model_{i}'
        
        PCCS = ('Epoch\tTime(sec)\tLoss_train\tLoss_val\tpearson\tMAE\tMSE\tRMSE\tr2')
        print(PCCS)

        with open(file_loss_min_PCCS, 'w') as f:
            f.write(PCCS + '\n')
        with open(file_best_pcc_PCCS, 'w') as f:
            f.write(PCCS + '\n')
        with open(file_best_r2_PCCS, 'w') as f:
            f.write(PCCS + '\n')

        start = timeit.default_timer()
        
        early_stopping = EarlyStopping(patience=7, verbose=True)
        for epoch in range(1, iteration + 1):

            print('Epoch:',epoch)
            loss_train_fold, y_train_true, y_train_predict = trainer.train(dataset_train, device, epoch, i, f'{analysis_dir}/fold{i}')
            pccs_val, mae_val, mse_val, rmse_val, r2_val, loss_val_fold, y_val_true, y_val_predict = tester.test(dataset_val, epoch, i, f'{analysis_dir}/fold{i}')

            end = timeit.default_timer()
            time = end - start
            
            early_stopping(loss_val_fold.tolist(), model, file_loss_min_model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            PCCS = [epoch, time, loss_train_fold.tolist(), loss_val_fold.tolist(), pccs_val.tolist(), mae_val, mse_val, rmse_val, r2_val]
            
            if loss_val_fold.tolist() < minloss:
                tester.save_pccs(PCCS, file_loss_min_PCCS)
                tester.save_model(model, file_loss_min_model)
                minloss = loss_val_fold.tolist()
            
            if pccs_val.tolist() > best_pearson:
                tester.save_pccs(PCCS, file_best_pcc_PCCS)
                tester.save_model(model, file_best_pcc_model)
                best_pearson = pccs_val.tolist()
            
            if r2_val > best_r2:
                tester.save_pccs(PCCS, file_best_r2_PCCS)
                tester.save_model(model, file_best_r2_model)
                best_r2 = r2_val
    
            print('\t'.join(map(str, PCCS)))