#!/usr/bin/env python
# coding: utf-8

# Reference: Molecular Similarity-Based Domain Applicability Metric Efficiently Identifies Out-of-Domain Compounds

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import oddt
from oddt.fingerprints import ECFP
import numpy as np
import sys

# In[30]:


from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

smiles_file = sys.argv[1]
plot_file = sys.argv[2]

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class AD:
    def __init__(self, train_data):
        if not isinstance(train_data, list) or not train_data or len(train_data) < 10:
            raise Exception("Training data should be a list of at least 10 SMILES")
        
        # Canonicalize the SMILES
        self.train_data = [Chem.CanonSmiles(sm) for sm in train_data if len(sm) > 0]
        self.fingerprints = []
    
    def fit(self):
        # Find the fingerprints
        for sm in self.train_data:
            mol = Chem.MolFromSmiles(sm)
            if not mol:
                continue
            self.fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2))
            
        if len(self.fingerprints) < 5:
            raise Exception(f"Not enough fingerprints can be generated from the training data (len={len(self.fingerprints)})")
    
    def get_score(self, smiles):
        if not self.fingerprints:
            raise Exception("Please run fit() first.")
            
        smiles = Chem.CanonSmiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        
        if not mol:
            raise Exception("Invalid SMILES.")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2)
        scores = []
        
        for train_fp in self.fingerprints:
            scores.append(DataStructs.TanimotoSimilarity(train_fp, fp))
        
        scores = np.array(scores)
        return np.sum(np.exp(-1 * (3 * scores) / (1 - scores)))


# ## Example

# In[18]:


import pandas as pd


# In[107]:


#smiles_file = 'data/3CL/3CL-balanced_randomsplit7_70_15_15_'
train_smiles = pd.read_csv(smiles_file + 'tr.csv', usecols=[4, 161])
test_smiles = pd.read_csv(smiles_file + 'te.csv', usecols = [4, 161])
valid_smiles = pd.read_csv(smiles_file + 'va.csv', usecols = [4, 161])


# In[108]:


result_df = pd.read_csv("../redial-2020/redial-2020-notebook-work/valid_test_csv_pred_cons/test/3CL_te.csv")


# In[109]:


tp_mask = result_df.apply(lambda x: x['Label'] == 1 and x['fingerprint_pred_label'] == 1, axis = 1)
tn_mask = result_df.apply(lambda x: x['Label'] == 0 and x['fingerprint_pred_label'] == 0, axis = 1)
fp_mask = result_df.apply(lambda x: x['Label'] == 0 and x['fingerprint_pred_label'] == 1, axis = 1)
fn_mask = result_df.apply(lambda x: x['Label'] == 1 and x['fingerprint_pred_label'] == 0, axis = 1)



# In[111]:


test_smiles = test_smiles[test_smiles.SMILES.isin(result_df.SMILES)]


# In[112]:


train_smiles_actives = train_smiles.Label.apply(lambda x: True if x == 1 else False)
test_smiles_actives = test_smiles.Label.apply(lambda x: True if x == 1 else False)
valid_smiles_actives = valid_smiles.Label.apply(lambda x: True if x == 1 else False)


# In[113]:


ad = AD(train_data=train_smiles.SMILES.to_list())
ad.fit()


# In[115]:


sdc_scores_test = np.array(test_smiles.SMILES.apply(lambda x: ad.get_score(x)))
sdc_scores_train = np.array(train_smiles.SMILES.apply(lambda x: ad.get_score(x)))
sdc_scores_valid = np.array(valid_smiles.SMILES.apply(lambda x: ad.get_score(x)))


# In[116]:


train_actives = sdc_scores_train[train_smiles_actives]
train_nactives = sdc_scores_train[~train_smiles_actives]
test_actives = sdc_scores_test[test_smiles_actives]
test_nactives = sdc_scores_test[~test_smiles_actives]
valid_actives = sdc_scores_valid[valid_smiles_actives]
valid_nactives = sdc_scores_valid[~valid_smiles_actives]


# In[117]:


result_df['sdc'] = result_df.SMILES.apply(lambda x: ad.get_score(x))

result_df['preds'] = ''
result_df['preds'][fn_mask] = 'fn'
result_df['preds'][fp_mask] = 'fp'
result_df['preds'][tn_mask] = 'tn'
result_df['preds'][tp_mask] = 'tp'
result_df['preds'][fn_mask] = 'fn'


# Get the frequency, PDF and CDF for each value in the series (Correct predictions)

s = result_df[tp_mask | tn_mask]['sdc']
df = pd.DataFrame(s)
stats_tdf = df.groupby('sdc')['sdc'].agg('count').pipe(pd.DataFrame).rename(columns = {'sdc': 'frequency'})

# PDF
stats_tdf['pdf'] = stats_tdf['frequency'] / sum(stats_tdf['frequency'])

# CDF
stats_tdf['cdf'] = stats_tdf['pdf'].cumsum()
stats_tdf = stats_tdf.reset_index()

# Get the frequency, PDF and CDF for each value in the series (Wrong predictions)

s = result_df[fp_mask | fn_mask]['sdc']
df = pd.DataFrame(s)
stats_fdf = df.groupby('sdc')['sdc'].agg('count').pipe(pd.DataFrame).rename(columns = {'sdc': 'frequency'})

# PDF
stats_fdf['pdf'] = stats_fdf['frequency'] / sum(stats_fdf['frequency'])

# CDF
stats_fdf['cdf'] = stats_fdf['pdf'].cumsum()
stats_fdf = stats_fdf.reset_index()

fig, axs = plt.subplots(figsize = (15, 10))
stats_tdf.plot(x = 'sdc', y = ['cdf'], grid = True, label = ['True'], ax = axs)
stats_fdf.plot(x = 'sdc', y = ['cdf'], grid = True, label = ['False'], ax = axs)
plt.legend()
plt.savefig(plot_file, dpi = 300)
