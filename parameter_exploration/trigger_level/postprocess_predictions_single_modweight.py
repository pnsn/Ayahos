import numpy as np
import os, sys, pandas, glob, time
import matplotlib.pyplot as plt
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


def plot_score_summary(df, x='thresh', y='f1', color='channel', line='model', subplot='weight'):
    plt.figure()
    nsubplots = len(df[subplot].unique())
    nrow = int(np.ceil(nsubplots**0.5))
    ncol = int(np.floor(nsubplots**0.5))+1
    if nrow != nsubplots**0.5:
        legend_insert = nsubplots//ncol
    nlines = len(df[line].unique())
    line_list = ['-','--','-.',':','*-','v-','^-']
    line_dict = dict(zip(df[line].unique(), line_list[:nlines]))
    for _i, sp in enumerate(df[subplot].unique()):
        plt.subplot(int(np.ceil(nsubplots**0.5)), int(np.floor(nsubplots**0.5))+1, _i+1)
        plt.title(sp)
        for col in df[(df[subplot]==sp)][color].unique():
            for _j, ln in enumerate(df[(df[subplot]==sp)&(df[color]==col)][line]):
                ls = line_dict[ln]
                xv = df[(df[subplot]==sp)&(df[line]==ln)&(df[color]==col)][x]
                yv = df[(df[subplot]==sp)&(df[line]==ln)&(df[color]==col)][y]
                if _j == 0:
                    plt.plot(xv, yv, ls, label=f'{ln} {col}')
                    plt.xlabel(x)
                    plt.ylabel(y)
                else:
                    plt.plot(xv, yv, ls)
        # plt.legend()


PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PRED = os.path.join(PROOT,'processed_data','avg_pred_stacks')
INPUT = os.path.join(PRED,'initial_trigger_assessment_summary_qt*.csv')
inputs = glob.glob(INPUT)
inputs.sort()
# df_in = pandas.read_csv(INPUT)
summary = []
cols = ['model','weight','channel','nele','narid','thresh','true_positive','false_positive','false_negative','precision','recall','f1']
tick = time.time()
for input in inputs:
    df_in = pandas.read_csv(input)
    qt = df_in.thresh.unique()[0]
    print(f'running threshold P>={qt:.2f} | elapsed time {(time.time() - tick)/60:.2f} min')
    multi_index = df_in[['model','weight','chan']].value_counts().index
    holder = []
    for model, weight, chan in multi_index:
        idf = df_in[(df_in.model == model)&(df_in.weight==weight)&(df_in.chan==chan)]
        cdf = parse_confusion_params(idf)
        tp = cdf.true_pos.sum()
        fp = cdf.false_pos.sum()
        fn = cdf.false_neg.sum()
        precision, recall, f1score = get_scores(tp, fp, fn)
        line = [model, weight, chan,
                len(idf), len(idf.arid.notna()),
                qt, tp, fp, fn,
                precision, recall, f1score]
        holder.append(line)
        summary.append(line)
    df_q = pandas.DataFrame(holder, columns=cols)
    df_q.to_csv(os.path.join(PRED,f'scored_trigger_assessment_summary_qt{qt:.2f}.csv'), header=True, index=False)

df_s = pandas.DataFrame(summary, columns=cols)
df_s.to_csv(os.path.join(PRED,'scored_trigger_assessment_summary_all.csv'), header=True, index=False)


# # Iterate across thresholds
# multi_index = df_in[['thresh','model','comp']]
# for qt, model, weight, comp in multi_index:
#     for label in df_


# tick = time.time()
# holder = []
# for _i, evid in enumerate(evid_list):
#     print(f'processing {evid} | {_i + 1:03}/{len(evid_list):03} | elapsed time: {(time.time() - tick)/60:.2f} min')
#     df = pandas.read_csv(eldstr.format(evid=evid))
#     # breakpoint()
#     df_tfpn = parse_confusion_params(df)
#     df = pandas.concat([df, df_tfpn], axis=1, ignore_index=False)
#     for model, weight, comp, thresh in df[['model','weight','comp','thresh']].value_counts().index:
#         idf = df[(df.model==model)&(df.weight==weight)&(df.comp==comp)]
#         tp = idf.true_pos.sum()
#         fp = idf.false_pos.sum()
#         fn = idf.false_neg.sum()
        
#         prec, rec, f1 = get_scores(tp, fp, fn)
#         line = [evid, model, weight, comp, thresh,
#                 len(idf.arid), len(idf.arid.notna()),
#                 tp, fp, fn, prec, rec, f1]
#         holder.append(line)
# sum_cols=['evid','model','weight','label','Pthresh',
#         'nlabeled','ntotal','true_positive','false_positive', 'false_negative',
#         'precision','recall','f1score']
# df_summary = pandas.DataFrame(holder, columns=sum_cols)
# df_summary.to_csv(sumsvstr, header=True, index=False)  