## Reading and Manipulating the dataset
from csv import DictReader

with open('datasets/drugs.csv', 'r') as csv_file:
    reader = DictReader(csv_file)
    compounds = [r for r in reader]
csv_file.close()

print(len(compounds))
for i in range(5):
    print(compounds[i])

## Encoding the smiles into computer understandable-numeric arrays
import smiles_encoder

## For canonical smiles
smiles_canonical_strings = [c['Canonical_SMILES'] for c in compounds]
encoder_canonical = smiles_encoder.SmilesEncoder(smiles_canonical_strings)
print(f'Number of unique dictionary elements: {len(encoder_canonical.element_dict.keys())}')
encoded_canonical_smiles = encoder_canonical.encode_many(smiles_canonical_strings)

## For isomeric smiles
smiles_isomeric_strings = [c['Isomeric_SMILES'] for c in compounds]
encoder_isomeric = smiles_encoder.SmilesEncoder(smiles_isomeric_strings)
print(f'Number of unique dictionary elements: {len(encoder_isomeric.element_dict.keys())}')
encoded_isomeric_smiles = encoder_isomeric.encode_many(smiles_isomeric_strings)

## Creating Smiles Datasets for further modeling
import pandas as pd
df_canonical = pd.DataFrame(encoded_canonical_smiles)
df_isomeric = pd.DataFrame(encoded_isomeric_smiles)

drug_indices_ds = pd.read_csv('datasets/drugs.csv')

df_canonical = df_canonical.add_prefix('canonical_')
df_isomeric = df_isomeric.add_prefix('isomeric_')

df_canonical['Drug_Index'] = drug_indices_ds['Drug_Index']
df_isomeric['Drug_Index'] = drug_indices_ds['Drug_Index']

df_canonical.to_csv('smiles_dataset/canonical_smiles.csv', index=False)
df_isomeric.to_csv('smiles_dataset/isomeric_smiles.csv', index=False)