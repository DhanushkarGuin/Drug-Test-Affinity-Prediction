import numpy as np

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: idx+1 for idx, aa in enumerate(amino_acids)}

import pandas as pd
df = pd.read_csv('datasets/proteins.csv')
df['sequence_length'] = df['Sequence'].apply(len)
max_length = df['sequence_length'].max()
print(f"The length of the longest sequence is: {max_length}")

MAX_LEN = 2549
sequences = df['Sequence'].tolist()

def int_encode_sequence(seq, max_len=MAX_LEN):
    seq_ids = [aa_to_int.get(aa, 0) for aa in seq[:max_len]]
    seq_ids += [0] * (max_len - len(seq_ids))
    return seq_ids

int_encoded_sequences = np.array([int_encode_sequence(seq, MAX_LEN) for seq in sequences])

encoded_sequences = pd.DataFrame(int_encoded_sequences)
encoded_sequences = encoded_sequences.add_prefix('sequence_')

encoded_sequences['Protein_Index'] = df['Protein_Index']
encoded_sequences.to_csv('sequence_dataset/sequences.csv',index=False)