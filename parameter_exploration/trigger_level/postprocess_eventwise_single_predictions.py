import numpy as np
import os, sys, pandas, glob, time
import matplotlib
ROOT = os.path.join('..','..')


def get_scores(tp, fp, fn):
    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp/ (tp + fp)
    if tp == 0 and fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = precision*recall / (precision + recall)
    return precision, recall, f1

def parse_confusion_params(df):
    holder = []
    for index, row in df.iterrows():
        # print(f'{index}/{len(df) - 1}')
        # If there is an analyst pick for this event/site/label
        if np.isfinite(row.arid):
            ## True, \hat{True} ##
            # If the trigger did capture the pick time
            if row.has_pick:
                tp = 1
                fn = 0
                fp = row.trigger_ct - 1
            # If the trigger did not capture the pick time
            else:
                tp = 0
                fn = 1
                fp = row.trigger_ct
        else:
            tp = 0
            fn = 0
            fp = row.trigger_ct
        holder.append([tp, fp, fn])
    out = pandas.DataFrame(holder, columns=['true_pos','false_pos','false_neg'], index=df.index)
    return out


PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PRED = os.path.join(PROOT,'processed_data')
eldstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','initial_trigger_assessment.csv')
# esvstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','initial_trigger_assessment_scored.csv')
sumsvstr = os.path.join(PRED,'avg_pred_stacks','initial_trigger_assessment_scored_summary.csv')
evid_list = glob.glob(os.path.join(PRED, 'avg_pred_stacks','uw*'))
evid_list.sort()
evid_list = [int(os.path.split(evid)[-1][2:]) for evid in evid_list]

tick = time.time()
holder = []
for _i, evid in enumerate(evid_list):
    print(f'processing {evid} | {_i + 1:03}/{len(evid_list):03} | elapsed time: {(time.time() - tick)/60:.2f} min')
    df = pandas.read_csv(eldstr.format(evid=evid))
    # breakpoint()
    df_tfpn = parse_confusion_params(df)
    df = pandas.concat([df, df_tfpn], axis=1, ignore_index=False)
    for model, weight, comp, thresh in df[['model','weight','comp','thresh']].value_counts().index:
        idf = df[(df.model==model)&(df.weight==weight)&(df.comp==comp)]
        tp = idf.true_pos.sum()
        fp = idf.false_pos.sum()
        fn = idf.false_neg.sum()
        
        prec, rec, f1 = get_scores(tp, fp, fn)
        line = [evid, model, weight, comp, thresh,
                len(idf.arid), len(idf.arid.notna()),
                tp, fp, fn, prec, rec, f1]
        holder.append(line)
sum_cols=['evid','model','weight','label','Pthresh',
        'nlabeled','ntotal','true_positive','false_positive', 'false_negative',
        'precision','recall','f1score']
df_summary = pandas.DataFrame(holder, columns=sum_cols)
df_summary.to_csv(sumsvstr, header=True, index=False)  