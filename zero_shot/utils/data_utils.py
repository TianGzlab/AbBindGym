import os

import numpy as np
import pandas as pd

from utils.scoring_utils import get_mutated_sequence


def DMS_file_cleanup(
    DMS_filename,
    target_seq,
    start_idx=1,
    end_idx=None,
    DMS_mutant_column="mutant",
    DMS_phenotype_name="score",
    DMS_directionality=1,
    AA_vocab="ACDEFGHIKLMNPQRSTVWY",
):
    """
    Borrowed from the Tranception codebase: https://github.com/OATML-Markslab/Tranception/blob/main/tranception/utils/dms_utils.py
    Function to process the raw substitution DMS assay data (eg., removing invalid mutants, aggregate silent mutations).
    """
    DMS_data = pd.read_csv(DMS_filename, low_memory=False)
    end_idx = start_idx + len(target_seq) - 1 if end_idx is None else end_idx
    DMS_data["mutant"] = DMS_data[DMS_mutant_column]

    DMS_data = DMS_data[DMS_data["mutant"].notnull()].copy()
    DMS_data = DMS_data[
        DMS_data["mutant"].apply(lambda x: all([len(y) >= 3 for y in x.split(":")]))
    ].copy()  # Mutant triplets should have at least 3 or more characters
    DMS_data = DMS_data[
        DMS_data["mutant"].apply(
            lambda x: all(
                [
                    (y[0] in AA_vocab) and (y[1:-1].isnumeric()) and (y[-1] in AA_vocab)
                    for y in x.split(":")
                ]
            )
        )
    ].copy()
    DMS_data = DMS_data[
        DMS_data["mutant"].apply(
            lambda x: all(
                [
                    int(y[1:-1]) - start_idx >= 0 and int(y[1:-1]) <= end_idx
                    for y in x.split(":")
                ]
            )
        )
    ].copy()
    DMS_data = DMS_data[
        DMS_data["mutant"].apply(
            lambda x: all(
                [y[0] == target_seq[int(y[1:-1]) - start_idx] for y in x.split(":")]
            )
        )
    ].copy()

    DMS_data[DMS_phenotype_name] = pd.to_numeric(
        DMS_data[DMS_phenotype_name], errors="coerce"
    )
    DMS_data = DMS_data[np.isfinite(DMS_data[DMS_phenotype_name])]
    DMS_data.dropna(subset=[DMS_phenotype_name], inplace=True)
    DMS_data["DMS_score"] = DMS_data[DMS_phenotype_name] * DMS_directionality
    DMS_data = DMS_data[["mutant", "DMS_score"]]
    DMS_data = DMS_data.groupby("mutant").mean().reset_index()

    DMS_data["mutated_sequence"] = DMS_data["mutant"].apply(
        lambda x: get_mutated_sequence(target_seq, x)
    )
    DMS_data = DMS_data[["mutant", "mutated_sequence", "DMS_score"]]

    return DMS_data


"""
def DMS_file_for_LLM(df):
    df['chain_id'] = df['chain_id'].fillna('')
    df['mutant'] = df['mutant'].apply(eval)
    df['mutated_sequence'] = df['mutated_sequence'].apply(eval)
    input_wt_seqs = []
    input_mt_seqs = []
    input_mutants = []
    for i in df.index:
        chain_ids = df.loc[i,'chain_id']
        wt_seqs = ''
        mt_seqs = ''
        mt_seq_dic = df.loc[i,'mutated_sequence']
        mutants = df.loc[i,'mutant']
        revise_mutants = []
        start_idx = 0
        for i,chain_id in enumerate(chain_ids):
            ms = mutants[chain_id]
            seq = list(mt_seq_dic[chain_id])
            for m in ms.split(':'):
                seq[int(m[1:-1])-1] = m[0]
                pos = int(m[1:-1]) + start_idx
                revise_mutants.append(m[:1]+str(pos)+m[-1:])
            wt_seqs += ''.join(seq)
            mt_seqs += mt_seq_dic[chain_id]
            start_idx += len(mt_seq_dic[chain_id])

        input_wt_seqs.append(wt_seqs)
        input_mt_seqs.append(mt_seqs)
        input_mutants.append(':'.join(revise_mutants))
    df['wildtype_sequence'] = input_wt_seqs
    df['mutated_sequence'] = input_mt_seqs
    df['mutant'] = input_mutants
    return df
"""


def DMS_file_for_LLM(df, focus=False, return_focus_chains=False):
    df["chain_id"] = df["chain_id"].fillna("")
    df["wildtype_sequence"] = df["wildtype_sequence"].apply(eval)
    df["mutant"] = df["mutant"].apply(eval)
    df["mutated_sequence"] = df["mutated_sequence"].apply(eval)
    input_wt_seqs = []
    input_mt_seqs = []
    input_focus_wt_seqs = []
    input_focus_mt_seqs = []
    input_mutants = []
    input_focus_mutants = []
    focus_chains = []
    for i in df.index:
        mutants = df.loc[i, "mutant"]
        for c in mutants:
            if c not in focus_chains:
                if mutants[c] != "":
                    focus_chains.append(c)
    for i in df.index:
        chain_ids = df.loc[i, "chain_id"]
        wt_seqs = ""
        mt_seqs = ""
        focus_wt_seqs = ""
        focus_mt_seqs = ""
        wt_seq_dic = df.loc[i, "wildtype_sequence"]
        mt_seq_dic = df.loc[i, "mutated_sequence"]
        mutants = df.loc[i, "mutant"]
        revise_mutants = []
        focus_revise_mutants = []
        start_idx = 0
        focus_start_idx = 0
        for i, chain_id in enumerate(chain_ids):
            ms = mutants[chain_id]
            if ms != "":
                for m in ms.split(":"):
                    pos = int(m[1:-1]) + start_idx
                    revise_mutants.append(m[:1] + str(pos) + m[-1:])
            wt_seqs += wt_seq_dic[chain_id]
            mt_seqs += mt_seq_dic[chain_id]
            start_idx += len(wt_seq_dic[chain_id])
            if chain_id in focus_chains:
                if ms != "":
                    for m in ms.split(":"):
                        pos = int(m[1:-1]) + focus_start_idx
                        focus_revise_mutants.append(m[:1] + str(pos) + m[-1:])
                focus_wt_seqs += wt_seq_dic[chain_id]
                focus_mt_seqs += mt_seq_dic[chain_id]
                focus_start_idx += len(wt_seq_dic[chain_id])

        input_wt_seqs.append(wt_seqs)
        input_mt_seqs.append(mt_seqs)
        input_mutants.append(":".join(revise_mutants))

        input_focus_wt_seqs.append(focus_wt_seqs)
        input_focus_mt_seqs.append(focus_mt_seqs)
        input_focus_mutants.append(":".join(focus_revise_mutants))
    if not focus:
        df["wildtype_sequence"] = input_wt_seqs
        df["mutated_sequence"] = input_mt_seqs
        df["mutant"] = input_mutants
    else:
        df["wildtype_sequence"] = input_focus_wt_seqs
        df["mutated_sequence"] = input_focus_mt_seqs
        df["mutant"] = input_focus_mutants
    if return_focus_chains:
        return df, sorted(focus_chains)
    return df


def generate_msa(df, msa_path, a2m_script_path, msa_db_path):
    for seq, g in df.groupby("wildtype_sequence"):
        name = g["POI"].values[0].split("_")[0] + g["chain_id"].values[0]
        if os.path.exists(f"{msa_path}/{name}.a2m"):
            continue
        with open(f"{msa_path}/{name}.fasta", "w") as f:
            f.write(f">{name}\n")
            f.write(f"{seq}\n")
        cmd = f"""bash {a2m_script_path} {msa_path} {name} 0.5 5 {msa_db_path}"""
        os.system(cmd)
