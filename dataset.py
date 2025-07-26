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

## Exporting the finalized dataset
final_ds.to_csv('Finalized_dataset.csv', index=False)