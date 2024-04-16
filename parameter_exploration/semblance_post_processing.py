import os, sys, pandas
import numpy as np

def split_reorder(label, delimiter='|'):
    parts = label.split('|')
    parts.sort()
    out = '|'.join(parts)
    return out

def sumrow(row):
    return row[['TP','FP','XP','TN','FN']].sum()

def calc_f1(row, xp_true=False):
    if xp_true:
        prec = (row.TP + row.XP)/(row.TP + row.XP + row.FP)
        rec = (row.TP + row.XP)/(row.TP + row.XP + row.FN)
    else:
        prec = (row.TP)/(row.TP + row.XP + row.FP)
        rec = (row.TP)/(row.TP + row.XP + row.FN)
    f1 = (2*prec*rec)/(prec + rec)
    return f1

def calc_mae(deltas, sampling_rate=100):
    mae = sum(deltas/sampling_rate)/len(deltas)
    return mae


ROOT = os.path.join('..','..')
PFILE = os.path.join('/Users','nates','Documents','Conferences',
                     '2024_SSA','PNSN','processed_data',
                     'avg_pred_stacks','semblance_trigger_scan_bulk.csv')

df = pandas.read_csv(PFILE)
df.model = df.model.apply(lambda x: split_reorder(x))
df.weight = df.weight.apply(lambda x: split_reorder(x))

df_conf = df.pivot_table(values=['TP','XP','FP','TN','FN'],
                         index=['component','thresh','weight','model','channel'],
                         aggfunc='sum')
# f1 calculation
df_conf = df_conf.assign(f1=lambda x: calc_f1(x))
# Get number of observations
df_conf = df_conf.assign(nobs=df_conf[['TP','XP','FP','TN','FN']].sum(axis=1))
# MAE calculation
mae_holder = []
for multi_index in df_conf.index:
    idf = df.copy()
    for sub_index, name in zip(multi_index, df_conf.index.names):
        idf = idf[idf[name] == sub_index]
    idf = idf[idf.nearest_peak_dsamp.notna()]
    mae = np.nansum(idf.nearest_peak_dsamp)*100/len(idf)
    mae_holder.append(mae)
df_conf = df_conf.assign(mae=mae_holder)