from dataclasses import dataclass
import numpy as np
import torch
from extra_utils import paired_msa_numbering, unpaired_msa_numbering, create_alignment

@dataclass
class aligned_results:
    aligned_seqs: list
    aligned_embeds: np.ndarray
    number_alignment: list

class AbAlignment:
    def __init__(self, device='cpu', ncpu=1):
        self.device = device
        self.ncpu = ncpu
        
    def number_sequences(self, seqs, chain='H', fragmented=False):
        if chain == 'HL':
            numbered_seqs, seqs, number_alignment = paired_msa_numbering(seqs, fragmented=fragmented, n_jobs=self.ncpu)
        else:
            numbered_seqs, seqs, number_alignment = unpaired_msa_numbering(seqs, chain=chain, fragmented=fragmented, n_jobs=self.ncpu)
        return numbered_seqs, seqs, number_alignment
    
    def align_encodings(self, encodings, numbered_seqs, seqs, number_alignment):
        aligned_encodings = []
        for res_embed, numbered_seq, seq in zip(encodings, numbered_seqs, seqs):
            aligned_encodings.append(create_alignment(res_embed, numbered_seq, seq, number_alignment))
        return np.concatenate([aligned_encodings], axis=0)
        
    def reformat_subsets(self, subset_list, mode='seqcoding', align=False, numbered_seqs=None, seqs=None, number_alignment=None):
        if mode in ['seqcoding', 'pseudo_log_likelihood', 'confidence']:
            return np.concatenate(subset_list)
        elif mode == 'restore' and align:
            # For restore mode with alignment, return the aligned sequences
            return subset_list[0] if len(subset_list) == 1 else subset_list
        elif mode == 'restore' and not align:
            # For restore mode without alignment, return the restored sequences
            return subset_list[0] if len(subset_list) == 1 else subset_list
        elif align:
            aligned_subsets = []
            for num, subset in enumerate(subset_list):
                start_idx = num * len(subset)
                end_idx = (num + 1) * len(subset)
                aligned_subset = self.align_encodings(
                    subset, 
                    numbered_seqs[start_idx:end_idx], 
                    seqs[start_idx:end_idx], 
                    number_alignment
                )
                aligned_subsets.append(aligned_subset)
            subset = np.concatenate(aligned_subsets)
            return aligned_results(
                aligned_seqs=[''.join(alist) for alist in subset[:,:,-1]],
                aligned_embeds=subset[:,:,:-1].astype(float),
                number_alignment=number_alignment.apply(lambda x: '{}{}'.format(*x[0]), axis=1).values
            )
        elif not align:
            return sum(subset_list, [])
        else:
            return np.concatenate(subset_list)
