import esm
import esm.inverse_folding
import torch
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing
import numpy as np
import os
from warnings import warn
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def get_available_chain(pdb_name):
    fpath = 'pdb/' + pdb_name + '.pdb'
    structure = esm.inverse_folding.util.load_structure(fpath)
    _, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    return list(native_seqs.keys())

def check_directory(output_folder):
    # Create directory if target directory not already exist
    path = os.path.join(os.getcwd(), output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
        print('Target directory created')

def check_pdb(fpath, pdb_name):
    # Check if pdb file exist
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'PDB file {pdb_name} not found')
    
def run_model(num_samples, fpath, temperature, output_folder, target_chain_id):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    
    highest = 0
    seqs = []
    recoveries = []
    for i in range(num_samples):
        coords, native_seqs = esm.inverse_folding.util.load_coords(fpath, target_chain_id)
        sampled_seq = model.sample(coords, temperature=temperature)
        seqs.append(sampled_seq)
        recovery = np.mean([(a==b) for a ,b in zip(native_seqs, sampled_seq)])
        recoveries.append(recovery)
        highest = max(recovery, highest)
    print(f'Best sequence recovery for chain {target_chain_id}:', highest)

    fname = output_folder + '/' + fpath.split('/')[1].split('.pdb')[0] + f'_{target_chain_id}' + '_seqs.fasta'
    with open(fname, 'w') as f:
        count = 1
        for seq in seqs:
            f.write(f'>: Sequence {count}, Recovery: {recoveries[count-1]}\n')
            f.write(f'{seq}\n')
            count += 1
    
    return [target_chain_id, highest]

def prediction_from_structure(output_folder, pdb_name, num_samples, target_chain_id_list=[], temperature:float=1)->None:
    """
    Save fasta files for predictions of each chain
    :parameter output_folder: name of the output directory, will create if not exist
    :parameter pdb_name: PDB ID, must be in pdb directory
    :parameter num_samples: number of predicting sequence generating
    :paremeter target_chain_id: list of chains to be predicted, if empty list provided, all chain would be used, might include antigen
    :parameter temperature: lower temperature, higher recovery but lower variation
    :return: None
    """
    check_directory(output_folder)
    
    fpath = 'pdb/' + pdb_name + '.pdb'
    check_pdb(fpath, pdb_name)
    
    cpu_count = multiprocessing.cpu_count()
    executor = ThreadPoolExecutor(max_workers=cpu_count)
    
    # Get all chain if nothing is specified
    if target_chain_id_list == []:
        target_chain_id_list = get_available_chain(pdb_name)
    
    futures = []
    for target_chain_id in target_chain_id_list:
        future = executor.submit(run_model, num_samples, fpath, temperature, output_folder, target_chain_id)
        futures.append(future)
    
    results = [future.result() for future in futures]
    result_dict = dict()
    for result in results:
        result_dict[result[0]] = result[1]
    
    executor.shutdown()
    
    return result_dict

def run_model_v2(num_samples, fpath, temperature, output_folder, target_chain_id):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    
    highest = 0
    seqs = []
    recoveries = []
    for i in range(num_samples):
        structure = esm.inverse_folding.util.load_structure(fpath, target_chain_id)
        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        sampled_seq = esm.inverse_folding.multichain_util.sample_sequence_in_complex(model ,coords, target_chain_id, temperature = temperature)
        seqs.append(sampled_seq)
        recovery = np.mean([(a==b) for a, b in zip(native_seqs[target_chain_id], sampled_seq)])
        recoveries.append(recovery)
        highest = max(recovery, highest)
    print(f'Best sequence recovery for chain {target_chain_id}:', highest)

    fname = output_folder + '/' + fpath.split('/')[1].split('.pdb')[0] + f'_{target_chain_id}' + '_seqs.fasta'
    with open(fname, 'w') as f:
        count = 1
        for seq in seqs:
            f.write(f'>: Sequence {count}, Recovery: {recoveries[count-1]}\n')
            f.write(f'{seq}\n')
            count += 1
    
    return target_chain_id, highest
    
    
def prediction_from_structure_v2(output_folder, pdb_name, num_samples, target_chain_id_list=[], temperature:float=1)->None:
    check_directory(output_folder)
    
    fpath = 'pdb/' + pdb_name + '.pdb'
    check_pdb(fpath, pdb_name)
    
    cpu_count = multiprocessing.cpu_count()
    executor = ThreadPoolExecutor(max_workers=cpu_count)
    
    futures = []
    for target_chain_id in target_chain_id_list:
        future = executor.submit(run_model_v2, num_samples, fpath, temperature, output_folder, target_chain_id)
        futures.append(future)
    
    results = [future.result() for future in futures]
    executor.shutdown()
    
    return results

def predict_mutational_effect(pdb_name, target_chain_id):
    _, native_seqs = get_available_chain(pdb_name)
    target_seq = native_seqs[target_chain_id]
    ll_fullseq, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(model, alphabet, coords, target_chain_id, target_seq, padding_length=10)
    print(f'Average log-likelihood on entire sequence: {ll_fullseq:.2f} (perplexity {np.exp(-ll_fullseq):.2f})')
    print(f'Average log-likelihood excluding missing coordinates: {ll_withcoord:.2f} (perplexity {np.exp(-ll_withcoord):.2f})')
    return ll_fullseq, ll_withcoord

    
    
    