# Import Functions and Files
from data_table import LightBBB, MoleculeNet, DeePred, B3BD
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt 
import pandas as pd 

# Variables
BBB_permeable_number = 0
BBB_nonpermeable_number = 0
invalid_molecule = []
molecular_weights = []

'''Find Number of SMILES Duplicate'''
data_table = pd.concat([LightBBB[['SMILES','BBclass']], MoleculeNet[['SMILES','BBclass']], DeePred[['SMILES','BBclass']], 
                        B3BD[['SMILES','BBclass']]], ignore_index=True) # Data Table with SMILES, BBclass
SMILES_count = data_table['SMILES'].value_counts() # Count the amount of times each SMILES occurs
duplicates = SMILES_count[SMILES_count > 1] # 1463 duplicate SMILES

'''Count the Number of Unique Compounds'''  
unique_compounds = SMILES_count[SMILES_count < 2] # 10635 unique SMILES

'''Make a Data Table with only unique SMILES'''
unique_SMILES_table = data_table.drop_duplicates(subset = 'SMILES', keep = 'first') # 12049 unique SMILES in total

'''Find # of BBB permeable and nonpermeable Drugs'''
for value in unique_SMILES_table['BBclass']: 
    if value == 1: 
        BBB_permeable_number = BBB_permeable_number + 1
    elif value == 0: 
        BBB_nonpermeable_number = BBB_nonpermeable_number + 1
print(BBB_permeable_number) # 8814 BBB+ molecules
print(BBB_nonpermeable_number) # 3284 BBB- molecules

'''Find Molecular Weight and remove invalid SMILES'''
for smiles in unique_SMILES_table['SMILES']:
    molecule = Chem.MolFromSmiles(smiles) # Convert SMILES to molecular name
    if molecule: 
        molecular_weight = Descriptors.MolWt(molecule) # Find the molecular weight for each SMILES
        molecular_weights.append(molecular_weight) # Add molecular weight to list
    else:
        invalid_molecule.append(smiles) # Add invalid SMILES to invalid_molecule  
unique_SMILES_table = unique_SMILES_table[~unique_SMILES_table['SMILES'].isin(invalid_molecule)] # Keep all SMILES that are valid
unique_SMILES_table['Molecular Weight (amu)'] = molecular_weights # Add molecular weights to Data Table

'''Make a Histogram of Molecular Weights'''
plt.hist(molecular_weights, edgecolor='black', align='mid')
ax = plt.gca()
ax.set_xlabel('Molecular Weight (amu)')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each Molecular Weight (amu) Category", size=11.5, weight='bold') 
plt.show()



