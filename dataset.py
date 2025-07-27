## Importing Necessary Datasets
import pandas as pd

ds_1 = pd.read_csv('datasets/drug_protein_affinity.csv')
ds_2 = pd.read_csv('datasets/drugs.csv')
ds_3 = pd.read_csv('datasets/proteins.csv')

## Merging datasets
ds_merged = pd.merge(ds_1,ds_2, on='Drug_Index')

final_ds = pd.merge(ds_merged,ds_3, on='Protein_Index')
print(final_ds.head())

## Checking for Missing Values if any
print(final_ds.isnull().sum())

## Importing the Smiles Dataset
canonical_df = pd.read_csv('smiles_dataset/canonical_smiles.csv')
isomeric_df = pd.read_csv('smiles_dataset/isomeric_smiles.csv')

## Merging with final dataset
final_ds = final_ds.drop(columns=['Canonical_SMILES','Isomeric_SMILES'])

final_ds = pd.merge(final_ds,canonical_df, on='Drug_Index')
final_ds = pd.merge(final_ds,isomeric_df, on='Drug_Index')

## Importing the Sequences dataset
sequences_df = pd.read_csv('sequence_dataset/sequences.csv')

## Merging with final dataset
final_ds = pd.merge(final_ds,sequences_df, on='Protein_Index')

## Dropping features that do not contribute to prediction
final_ds = final_ds.drop(columns=['Sequence','Drug_Index', 'Protein_Index', 'Accession_Number','Gene_Name','CID'])

## Exporting the finalized dataset
final_ds.to_csv('Finalized_dataset.csv', index=False)