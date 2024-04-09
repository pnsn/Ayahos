import os, pandas, glob, time
import matplotlib.pyplot as plt

PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PRED = os.path.join(PROOT,'processed_data','avg_pred_stacks')
INPUT = os.path.join(PRED,'initial_trigger_assessment_sum_v2_Pthresh_*_scored.csv')
inputs = glob.glob(INPUT)
inputs.sort()

def run_cmat_stats(ser_C):
    TP = ser_C.TP
    FP = ser_C.FP
    TN = ser_C.TN
    FN = ser_C.FN
    N = ser_C.sum()
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
    
    return {'err': err, 'acc': acc, 'pre': pre, 'rec': rec, 'f1': F1}

Pthresh = [float(inpt[-15:-11]) for inpt in inputs]
tf_dict = dict(zip(Pthresh, inputs))

pholder = {}
f1_threshold = 0.8
pshortlist = {}
tick = time.time()
for _k, _f in tf_dict.items():
    print(f'processing {_f} | et: {(time.time() - tick)/60:.2f} min')
    df = pandas.read_csv(_f)

    # For traces with data
    df_l = df[df.labeled_data]
    # Get general confusion matrix elements using bool elements
    ser_l_f = (df_l[['TP','FP','TN','FN']] > 0).sum()
    cstats_l = run_cmat_stats(ser_l_f)

    multi_index = df_l[['chan','model','weight']].value_counts().index
    holder = {'group_labeled': cstats_l}
    shortlist = {}
    for chan, model, weight in multi_index:
        idf_l = df_l[(df_l.chan==chan)&(df_l.model==model)&(df_l.weight==weight)]
        ser_il = (idf_l[['TP','FP','TN','FN']] > 0).sum()
        cstats_il = run_cmat_stats(ser_il)
        holder.update({(chan, model, weight): cstats_il})
        if cstats_il['f1'] >= f1_threshold:
            shortlist.update({(chan, model, weight): cstats_il})
    pholder.update({_k: holder.copy()})
    pshortlist.update({_k: shortlist.copy()})