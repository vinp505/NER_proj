"""Helper script to merge all evaluation csv files into one.
Mainly used for plotting purposes."""

import pandas as pd

# ------------------------------------------------------------

# store file prefix and all suffixes
prefix = 'evaluation_ftmodel_'
tl = ['all', 'chi', 'dan', 'eng', 'rom', 'slk']

# iterate through files, save the loaded df
df_list = []
for lang in tl:
    df = pd.read_csv(prefix + lang + '.csv')
    df['targ_lang'] = lang
    df = df.rename({'lang' : 'test_lang'}, axis= 1)
    df_list.append(df)

# concatenate all datasets
df_merged = pd.concat(df_list, axis= 0)

# UNUSED - group languages
def grouper(row):
    if row['test_lang'] in ['eng', 'ger', 'dan', 'nor', 'swe']:
        return 1
    elif row['test_lang'] in ['slk', 'srb', 'hrv']:
        return 2
    elif row['test_lang'] in ['rom', 'por']:
        return 3
    elif row['test_lang'] == 'heb':
        return 4
    elif row['test_lang'] == 'chi':
        return 5
    raise WindowsError("fuck YOU")
df_merged['group'] = df_merged.apply(grouper, axis= 1)

# save resulting dataframe
df_merged.to_csv("evaluation_merged.csv", index= False)
print("Merged DataFrame saved as [evaluation_merged.csv]")