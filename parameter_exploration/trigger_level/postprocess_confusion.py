import os, pandas, glob, time
import matplotlib.pyplot as plt
import numpy as np

PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PRED = os.path.join(PROOT,'processed_data','avg_pred_stacks')
INPUT = os.path.join(PRED,'scored_trigger_assessment','initial_trigger_assessment_sum_v2_Pthresh_*_scored.csv')
inputs = glob.glob(INPUT)
inputs.sort()

def run_cmat_stats(ser_C):
    TP = ser_C.TP
    FP = ser_C.FP
    TN = ser_C.TN
    FN = ser_C.FN
    N = ser_C.sum()
    out = dict(zip(['nTP','nFP','nTN','nFN','N'], [TP,FP,TN,FN,N]))
    err = (FP + FN)/N
    acc = (TP + TN)/N
    if TN == 0 and FP == 0:
        pre = 0
    else:
        pre = TP/(TN + FP)
    if TP == 0 and FN == 0:
        rec = 0
    else:
        rec = TP/(TP + FN)
    if TP == 0 and TN==0 and FP == 0:
        F1 = 0
    else:
        F1 = TP/(TP + (FN + FP)*0.5)
    out.update({'err': err, 'acc': acc, 'pre': pre, 'rec': rec, 'f1': F1})
    return out


def calc_pick_errors(df, sampling_rate=100):
    ser_npds = df[df.labeled_data].nearest_peak_dsmp
    ser_npdt = ser_npds/sampling_rate
    mae = np.abs(ser_npdt.sum())/len(ser_npdt)
    rmse = np.sqrt(np.sum(ser_npdt.values**2)/len(ser_npdt))
    pc = df[df.labeled_data].TP.sum()/len(df[df.labeled_data])
    u_res = ser_npdt.mean()
    return {'nlbl': len(ser_npds), 'MAE': mae, 'RMSE': rmse, 'PC': pc, 'u_res': u_res}

def run_f1_pick_err(df, sampling_rate=100, f1mode='bool'):
    out = {}
    out.update(calc_pick_errors(df, sampling_rate=sampling_rate))
    if f1mode=='bool':
        ser_C = (df[['TP','FP','TN','FN']] > 0).sum()
    elif f1mode == 'count':
        ser_C = df[['TP','FP','TN','FN']].sum()
    mod_stats = run_cmat_stats(ser_C)
    out.update(mod_stats)
    return out
    

Pthresh = [float(inpt[-15:-11]) for inpt in inputs]
tf_dict = dict(zip(Pthresh, inputs))
sampling_rate=100
multi_index_fields = ['chan','model','weight']

tick = time.time()
main_dict = {}
for _k, _f in tf_dict.items():
    if _k <= 0.1:
        continue
    df = pandas.read_csv(_f)
    df = df[df.labeled_data]

    # Split by Instrument Type
    df_mi = df.set_index(multi_index_fields)
    # df_l = df[df.labeled_data]
    holder = []
    for _mi in df_mi.index.unique():
        idf_mi = df_mi.loc[_mi]
        stats = run_f1_pick_err(idf_mi, sampling_rate=sampling_rate, f1mode='count')
        line = list(stats.values())
        holder.append(line)
    df_stats = pandas.DataFrame(holder, columns=list(stats.keys()), index=df_mi.index.unique())
    df_stats.to_csv(os.path.join(PRED,'performance_stats_LTO', f'pt{_k:.2}_{"_".join(multi_index_fields)}_STATS_LTO.csv'), header=True, index=True)
    main_dict.update({f'{_k} mi': df_stats.copy()})

    # # Split by seismic site
    # df_smwc = df.set_index(['net','sta','model','weight','comp'])
    # holder = []
    # for _smwc in df_smwc.index.unique():
    #     idf_smwc = df_smwc.loc[_smwc]
    #     stats = run_f1_pick_err(idf_smwc, sampling_rate=sampling_rate, f1mode='count')
    #     holder.append(list(stats.values()))
    # df_stats = pandas.DataFrame(holder, columns=list(stats.keys()), index=df_smwc.index.unique())
    # main_dict.update({f'{_k} smwc': df_stats.copy()})
    # df_stats.to_csv(os.path.join(PRED,f'pt{_k:.2}_net_sta_model_weight_comp_STATS_LTO.csv'), header=True, index=True)





    # print(f'processing {_f} | et: {(time.time() - tick)/60:.2f} min')
    # df = pandas.read_csv(_f)

    # # For traces with data
    # df_l = df[df.labeled_data]
    # # Get general confusion matrix elements using bool elements
    # ser_l_f = (df_l[['TP','FP','TN','FN']] > 0).sum()
    # cstats_l = run_cmat_stats(ser_l_f)

    # multi_index = df_l[['chan','model','weight']].value_counts().index
    # holder = {'group_labeled': cstats_l}
    # shortlist = {}
    # for chan, model, weight in multi_index:
    #     idf_l = df_l[(df_l.chan==chan)&(df_l.model==model)&(df_l.weight==weight)]
    #     ser_il = (idf_l[['TP','FP','TN','FN']] > 0).sum()
    #     cstats_il = run_cmat_stats(ser_il)
    #     holder.update({(chan, model, weight): cstats_il})
    #     if cstats_il['f1'] >= f1_threshold:
    #         shortlist.update({(chan, model, weight): cstats_il})
    # pholder.update({_k: holder.copy()})
    # pshortlist.update({_k: shortlist.copy()})