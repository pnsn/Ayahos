import os, sys, pandas, logging
import numpy as np
from itertools import chain, combinations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

Logger = logging.getLogger('semblance_post_processing')

def powerset(iterable, with_null=False):
    """
    Create a powerset from an iterable object comprising set elements

    scales as 2**len(iterable) (including the null set)
    
    {1, 2, 3} -> [(1,2,3), 
                  (1,2), (1,3), (2,3),
                  (1), (2), (3),
                  ()] <-- NOTE: null setexcluded if with_null=False

    :: INPUTS ::
    :param iterable: [lis] iterable set of elements from which
                        a power set is generated.
                    NOTE: iterable is passed through a list(set(iterable))
                        nest to ensure no duplicate entries
    :param with_null: [bool] should the null set be included

    :: OUTPUT ::
    :return out: [list] of [tuple] power set
    
    Source Attribution: User Mark Rushakoff and edits by User Ran Feldesh
    https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    """
    s = list(set(iterable))
    if with_null:
        out = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    else:
        out = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return out

def split_reorder(label, delimiter='|'):
    parts = label.split('|')
    parts.sort()
    out = '|'.join(parts)
    return out

def sumrow(row):
    return row[['TP','FP','XP','TN','FN']].sum()

def calc_prec(row, xp_true=False):
    if xp_true:
        prec = (row.TP + row.XP)/(row.TP + row.XP + row.FP)
    else:
        prec = (row.TP)/(row.TP + row.XP + row.FP)
    return prec

def calc_fpr(row, xp_true=False):
    if xp_true:
        fpr = (row.FP)/(row.FP + row.TN)
    else:
        fpr = (row.FP + row.XP)/(row.FP + row.XP + row.TN)
    return fpr

def calc_rec(row, xp_true=False):
    if xp_true:
        rec = (row.TP + row.XP)/(row.TP + row.XP + row.FN)
    else:
        rec = (row.TP)/(row.TP + row.XP + row.FN)
    return rec


def calc_f1(row, xp_true=False):
    if xp_true:
        prec = (row.TP + row.XP)/(row.TP + row.XP + row.FP)
        rec = (row.TP + row.XP)/(row.TP + row.XP + row.FN)
    else:
        prec = (row.TP)/(row.TP + row.XP + row.FP)
        rec = (row.TP)/(row.TP + row.XP + row.FN)
    f1 = (2*prec*rec)/(prec + rec)
    return f1


def get_subset_df(df_src, multi_index_entry, multi_index_names):
    idf = df_src
    for _i, name in enumerate(multi_index_names):
        idf = idf[getattr(idf, name)==multi_index_entry[_i]]
    return idf.copy()


def calc_mae_pc(df_src, multi_index, sampling_rate=100):
    holder = {'MAE': [], 'PC': []}
    # Iterate across multi-indices
    for _i, mi in enumerate(multi_index):
        Logger.debug(f'calc_mae_pc iteration {_i + 1}/{len(multi_index)}')
        # Get subset, correctly labeled metadata
        idf = get_subset_df(df_src, mi, multi_index.names)
        # Calculate PC as the sum of True Positives divided by sample size
        pc = len(idf[(idf.correctly_labeled)&(idf.pick_in_trigger)])/len(idf[idf.correctly_labeled])
        # Get pick offsets from labeled data 
        deltas = idf[idf.TP == 1].nearest_peak_dsamp
        # Convert to seconds
        dts = abs(deltas/sampling_rate)
        # Calculate MAE
        mae = sum(dts)/len(dts)
        # Collect iterated results
        holder['MAE'].append(mae)
        holder['PC'].append(pc)
    # Associate multi-index to derived values for output
    df_out = pandas.DataFrame(holder, index=multi_index)
    return df_out

def process_pivot(df,
                  universal_indices=['model','weight','component','thresh'],
                  test_indices=['channel'],
                  universal_values=['TP','XP','FP','TN','FN'],
                  sampling_rate=100):

    # Merge test and universal indices
    uindex = universal_indices + test_indices
    # Create pivot table
    dfp = df.pivot_table(values=universal_values, index=uindex, aggfunc='sum')
    # Get f1/prec/rec
    dfp = dfp.assign(precision=lambda x: calc_prec(x))
    dfp = dfp.assign(recall=lambda x: calc_rec(x))
    dfp = dfp.assign(FPR=lambda x: calc_fpr(x))
    # f1 calculation
    dfp = dfp.assign(f1=lambda x: calc_f1(x))
    # Get number of observations
    dfp = dfp.assign(nobs=dfp[['TP','XP','FP','TN','FN']].sum(axis=1))
    # Get MAE and PC
    dfmp = calc_mae_pc(df, dfp.index, sampling_rate=sampling_rate)
    df_out = pandas.concat([dfp, dfmp], axis=1, ignore_index=False)
    return df_out



PFILE = os.path.join('/Users','nates','Documents','Conferences',
                     '2024_SSA','PNSN','processed_data',
                     'avg_pred_stacks','semblance_trigger_scan_bulk.csv')
# Load data
df = pandas.read_csv(PFILE)
# Homogenize ordering of model and weight names
df.model = df.model.apply(lambda x: split_reorder(x))
df.weight = df.weight.apply(lambda x: split_reorder(x))

# Universal indices always used - model|weight|label(component)
uindices = ['model','weight','thresh']
tindices = ['component','channel']
uvalues = ['TP','XP','FP','TN','FN']

pset = powerset(tindices, with_null=True)
Logger.info(f'starting power-set iterations over a set of {len(tindices)} ({2**len(tindices)} sets)')
for iset in pset:
    Logger.info(f'processing test parameter set {iset}')
    df_out = process_pivot(df, universal_indices=uindices,
                           universal_values=uvalues,
                           test_indices=list(iset))
    out_path, _ = os.path.split(PFILE)
    out_fn = '__'.join(['semblance_trigger_scan'] + list(df_out.index.names))
    df_out.to_csv(os.path.join(out_path, f'{out_fn}.csv'), header=True, index=True)







    # if len(iset) > 0:
    #     for mindex in df[list(iset)].value_counts().index:
    #         idf_out = get_subset_df(df_out, mindex)



# df_conf = df.pivot_table(values=['TP','XP','FP','TN','FN'],
#                          index=['component','thresh','weight','model','channel'],
#                          aggfunc='sum')
# # f1 calculation
# df_conf = df_conf.assign(f1=lambda x: calc_f1(x))
# # Get number of observations
# df_conf = df_conf.assign(nobs=df_conf[['TP','XP','FP','TN','FN']].sum(axis=1))



# # MAE calculation
# mae_holder = []
# for multi_index in df_conf.index:
#     idf = df.copy()
#     for sub_index, name in zip(multi_index, df_conf.index.names):
#         idf = idf[idf[name] == sub_index]
#     idf = idf[idf.nearest_peak_dsamp.notna()]
#     mae = np.nansum(idf.nearest_peak_dsamp)*100/len(idf)
#     mae_holder.append(mae)
# df_conf = df_conf.assign(mae=mae_holder)