# Import Functions and Files
from data_table import LightBBB, MoleculeNet, DeePred, B3BD
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import numpy as np, matplotlib.pyplot as plt 
from scipy.stats import ttest_ind
from scipy import stats
import pandas as pd 

# Variables
BBB_permeable_number = 0
BBB_nonpermeable_number = 0
invalid_molecule = []
molecular_weights = []
logPs = []
TPSAs = []
BBB_permeable = []
BBB_nonpermeable = []
unique_multiple_BBB_compound = []
invalid_multiple_BBB_compound = []

'''Find Number of SMILES Duplicate'''
data_table = pd.concat([LightBBB[['SMILES','BBclass']], MoleculeNet[['SMILES','BBclass']], DeePred[['SMILES','BBclass']], 
                        B3BD[['SMILES','BBclass']]], ignore_index=True) # Data Table with SMILES, BBclass
SMILES_count = data_table['SMILES'].value_counts() # Count the amount of times each SMILES occurs
duplicates = SMILES_count[SMILES_count > 1] # 1463 duplicate SMILES

'''Count the Number of Unique Compounds'''  
unique_compounds = SMILES_count[SMILES_count < 2] # 10635 unique SMILES

'''Table of Duplicate Compounds and Reported BBclass Data'''
duplicate_SMILES = duplicates.index.to_numpy() # Makes a numpy array of all the duplicate SMILES
duplicate_SMILES_table = data_table[data_table['SMILES'].isin(duplicate_SMILES)] # Makes a table of every instance of duplicate SMILES 
duplicate_SMILES_table = duplicate_SMILES_table.groupby('SMILES')['BBclass'].nunique() # Makes a table that shows the number of unique reported values

'''Filtering Duplicates based on whether the reported values are consistent or not'''
for i in range(len(duplicate_SMILES_table)):
    if duplicate_SMILES_table.iloc[i] == 1: 
        unique_multiple_BBB_compound.append(duplicate_SMILES_table.index[i])
    else: 
        invalid_multiple_BBB_compound.append(duplicate_SMILES_table.index[i])
unique_SMILES_table = data_table[~data_table['SMILES'].isin(invalid_multiple_BBB_compound)]

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

'''Find logP value and remove invalid SMILES'''
for smiles in unique_SMILES_table['SMILES']:
    molecule = Chem.MolFromSmiles(smiles) # Convert SMILES to molecular name 
    logP = Crippen.MolLogP(molecule) # Find the logP value for each SMILES 
    logPs.append(logP) 
unique_SMILES_table['LogP Value'] = logPs # Add logP value to the Data Table

'''Find TPSA value and remove invalid SMILES'''
for smiles in unique_SMILES_table['SMILES']:
    molecule = Chem.MolFromSmiles(smiles) # Convert SMILES to molecular name 
    TPSA = rdMolDescriptors.CalcTPSA(molecule) # Find the TPSA value for each SMILES 
    TPSAs.append(TPSA) 
unique_SMILES_table['TPSA Value'] = TPSAs # Add TPSA value to the Data Table

'''Find # of BBB permeable and nonpermeable Drugs'''
for value in unique_SMILES_table['BBclass']: 
    if value == 1: 
        BBB_permeable_number = BBB_permeable_number + 1
    elif value == 0: 
        BBB_nonpermeable_number = BBB_nonpermeable_number + 1
print(BBB_permeable_number) # 9589 BBB+ molecules
print(BBB_nonpermeable_number) # 3601 BBB- molecules

'''Distribute SMILES into separate tables based on BBB permeability'''
BBB_permeable = unique_SMILES_table[unique_SMILES_table['BBclass'] == 1]['SMILES'].tolist()
BBB_nonpermeable = unique_SMILES_table[unique_SMILES_table['BBclass'] == 0]['SMILES'].tolist()
BBB_permeable_table = unique_SMILES_table[~unique_SMILES_table['SMILES'].isin(BBB_nonpermeable)] # Keep all BBB+ permeable molecules
BBB_nonpermeable_table = unique_SMILES_table[~unique_SMILES_table['SMILES'].isin(BBB_permeable)] # Keep all BBB+ nonpermeable molecules

'''Get TPSA Values for BBB+ and BBB- molecules individually'''
tpsa_positive = BBB_permeable_table['TPSA Value']
tpsa_negative = BBB_nonpermeable_table['TPSA Value']

'''Get logP Values for BBB+ and BBB- molecules individually'''
logP_positive = BBB_permeable_table['LogP Value']
logP_negative = BBB_nonpermeable_table['LogP Value']

'''Find P-value between TPSA of BBB+ and BBB- molecules'''
tpsa_positive_array = np.array(tpsa_positive)
tpsa_negative_array = np.array(tpsa_negative)
t_stat, p_value = ttest_ind(tpsa_positive_array, tpsa_negative_array)
print(f'The P-value between TPSA of BBB+ and BBB- molecules is {p_value}')

'''Find Confidence Interval of TPSA difference of BBB+ and BBB- molecules'''
n1 = len(tpsa_positive) # Size of the BBB+ dataset
n2 = len(tpsa_negative) # Size of the BBB- dataset
mean1 = np.mean(tpsa_positive_array)
mean2 = np.mean(tpsa_negative_array)
sd1 = np.std(tpsa_positive_array, ddof = 1) # Sample Standard Deviation of TPSA of BBB+ dataset
sd2 = np.std(tpsa_negative_array, ddof = 1) # Sample Standard Deviation of TPSA of BBB- dataset
pooled_standard_deviation = np.sqrt(((n1-1)*(sd1**2)+(n2-1)*(sd2**2))/(n1+n2-2))
standard_error = pooled_standard_deviation * np.sqrt((1/n1)+(1/n2))
s1 = np.var(tpsa_positive_array, ddof = 1) # Sample variance of TPSA of BBB+ dataset
s2 = np.var(tpsa_negative_array, ddof = 1) # Sample variance of TPSA of BBB- dataset
numerator = ((s1/n1)+ (s2/n2))**2
denominator = (((s1/n1)**2)/(n1-1))+ (((s2/n2)**2)/(n2-1))
degrees_of_freedom = numerator/denominator # Calculate degrees of freedom using the Welch-Satterthwaite formula for unequal variance
t_critical_value = stats.t.ppf(0.975, degrees_of_freedom) # Critical Value for 95% confidence 
critical_value_right = round(((mean1-mean2) + t_critical_value * np.sqrt((s1/n1)+(s2/n2))), 5) # + Critical Value
critical_value_left = round(((mean1-mean2) - t_critical_value * np.sqrt((s1/n1)+(s2/n2))), 5) # - Critical Value
print(critical_value_left)
print(critical_value_right)

'''Find Effect Size of TPSA difference of BBB+ and BBB- molecules'''
effective_size = (mean1 - mean2)/pooled_standard_deviation # Cohen's d for effective size
print(effective_size)


'''Find P-value between logP of BBB+ and BBB- molecules'''
logP_positive_array = np.array(logP_positive)
logP_negative_array = np.array(logP_negative)
t_stat, p_value = ttest_ind(logP_positive_array, logP_negative_array)
print(f'The P-value between logP of BBB+ and BBB- molecules is {p_value}')

'''Make a Histogram of Molecular Weights'''
bins = np.arange(min(molecular_weights), max(molecular_weights) + 1, 50)
plt.hist(molecular_weights, edgecolor='black', align='mid', bins = bins)
ax = plt.gca()
ax.set_xlabel('Molecular Weight (amu)')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each Molecular Weight (amu) Category", size=11.5, weight='bold') 
plt.show()

'''Make a Histogram of LogP Value'''
bins = np.arange(min(logPs), max(logPs) + 1, 1)
plt.hist(logPs, edgecolor ='black', align='mid', bins = bins)
ax = plt.gca()
ax.set_xlabel('LogP Value')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each LogP Value Category", size=13.5, weight='bold') 
plt.show()

'''Make a Histogram of the TPSA Value'''
plt.hist(logPs, edgecolor ='black', align='mid', bins = 50)
ax = plt.gca()
ax.set_xlabel('TPSA (angstrom)')
ax.set_ylabel('Number of Molecules')
ax.set_title("Number of Molecules in each TPSA (angstrom) Category", size=11.5, weight='bold') 
plt.show()

'''Make a Box Plot of the TPSA Value, BBB+ and BBB-'''
fig, ax = plt.subplots(1, 2)
ax[0].boxplot(tpsa_positive, vert = False, showfliers = False)
ax[0].set_title("TPSA Distribution for BBB+ molecules", size=11.5, weight='bold')
ax[0].set_xlabel("TPSA (angstrom)")
ax[1].boxplot(tpsa_negative, vert = False, showfliers = False)
ax[1].set_title("TPSA Distribution for BBB- molecules", size=11.5, weight='bold')
ax[1].set_xlabel("TPSA (angstrom)")
plt.show()

'''Make a Box Plot of the logP Value, BBB+ and BBB-'''
fig, ax = plt.subplots(1, 2)
ax[0].boxplot(logP_positive, vert = False, showfliers = False)
ax[0].set_title("logP Distribution for BBB+ molecules", size=11.5, weight='bold')
ax[0].set_xlabel("logP")
ax[1].boxplot(logP_negative, vert = False, showfliers = False)
ax[1].set_title("logP Distribution for BBB- molecules", size=11.5, weight='bold')
ax[1].set_xlabel("logP")
plt.show()



