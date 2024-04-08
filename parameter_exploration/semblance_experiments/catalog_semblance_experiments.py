import os, sys, glob, obspy, pandas
import numpy as np
CROOT = os.path.join('..','..')
sys.path.append(os.path.join(CROOT,'wyrm'))
from wyrm.data.dictstream import DictStream
from wyrm.data.mltrace import MLTrace
from wyrm.util.time import unix_to_epoch
import wyrm.util.feature_extraction as fex
import wyrm.util.stacking as stk

PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PICK = os.path.join(PROOT,'data','AQMS','AQMS_event_mag_phase_query_output.csv')
PRED = os.path.join(PROOT,'processed_data')

pfstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','{model}','{weight}','{site}','{file}')
# Load bulk as well...

mwd = {'EQTransformer': ['stead','pnw','iquique','lendb','instance'],
       'PhaseNet': ['stead','diting','iquique','lendb','instance']}

evid_list = glob.glob(os.path.join(PRED, 'avg_pred_stacks','uw*'))
evid_list.sort()
evid_list = [int(os.path.split(evid)[-1][2:]) for evid in evid_list]
pick_df = pandas.read_csv(PICK, index_col='arid')
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x: obspy.UTCDateTime(unix_to_epoch(float(x))))
pick_df = pick_df[pick_df.evid.isin(evid_list)]

def make_bkey(subset, source_set):
    string = ''
    for _i in range(len(source_set)):
        if _i in subset:
            string += '1'
        else:
            string += '0'

    return string


def process_det(row, mlt):
    line = []
    # Get maximum value of the data
    Pmax = np.nanmax(mlt.data)
    # Get the p05 quantile
    Pp05 = np.quantile(mlt.data, q=0.05)
    # Get the index of the pick time 
    ii = mlt.utcdatetime_to_nearest_index(row.arrdatetime)
    # Get the probability value at the pick time
    Pv = mlt.data[ii]
    # Get the fold at the pick time
    Pf = mlt.fold[ii]
    line = [mlt.id, arid, row.arrdatetime, Pmax, Pp05, ii, Pv, Pf]
    # run triggering on a series of probability thresholds
    for phat in [0.05, 0.1, 0.3, 0.5, 0.8, 0.9]:
        qtriggers = obspy.signal.trigger.trigger_onset(mlt.data, phat, phat)
        if len(qtriggers) == 0:
            has_pick = False
        else:
            for qt in qtriggers:
                if qt[0] <= ii <= qt[1]:
                    has_pick = True
                else:
                    has_pick = False
        line.append(len(qtriggers))
        line.append(has_pick)
    evid_ser = pandas.Series(line, index=['pred_id','arid','arrdatetime','Pmax','Pp05','ii','Pv','Pf',
                                          'Pt0.05_nt','Pt0.05_hp','Pt0.10_nt','Pt0.10_hp','Pt0.30_nt','Pt0.30_hp',
                                          'Pt0.50_nt','Pt0.50_hp','Pt0.80_nt','Pt0.80_hp','Pt0.90_nt','Pt0.90_hp'])
    evid_ser.name = evid_ser.pred_id
    return evid_ser

# Iterate across EVIDs
for evid in evid_list:
    print(f'processing evid uw{evid}')
    # Iterate across analyst picks from AQMS for uw{EVID}
    for arid, row in pick_df[pick_df.evid == evid].iterrows():
        # Get list of predictions for this site/phase combination
        pred_files = glob.glob(pfstr.format(evid=evid,
                                           model='*',
                                           weight='*',
                                           site=f'{row.net}.{row.sta}',
                                           file=f'*.*.*.??{row.iphase}.*'))
        mltlist = [MLTrace.read(pf) for pf in pred_files]
        dst = DictStream(traces = mltlist)
        dst.trim(starttime=dst.stats.max_starttime, endtime=dst.stats.min_endtime)
        pstack = np.c_[[mlt.data for mlt in dst]]
        fstack = np.c_[[mlt.fold for mlt in dst]]
        pick_holder = []
        # semb_full = semblance(pstack, fstack, **sembkwargs)
        # pick_holder.append(semb_full)
        # Generate powerset of prediction IDs
        powset = powerset(range(len(dst)))
        for iset in powset:
            bkey = make_bkey(iset, len(dst))
            ipstack = np.c_[[pstack[_i] for _i in iset]]
            ifstack = np.c_[[fstack[_i] for _i in iset]]
            breakpoint()
            # semb_set = semblance(pstack, fstack, **sembkwargs)

            

            
    #     for iset in powset:

    #     for _id, mlt in dst.traces.items():
    #         # run individual assessment
    #         result = process_det(row, mlt):
    #         # run LOOCV assessment

    #         # run 
    #     # Run loocv
    #     for _id in dst.traces.keys():
    #         semb_mlt = semblance()
    # breakpoint()
                

            

        # for tr in dst:

    # pred_file_list = glob.glob(fstr.format(evid=evid, model='*', weight='*', site='*'))
