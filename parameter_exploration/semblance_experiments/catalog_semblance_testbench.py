import os, sys, glob, obspy, pandas
import numpy as np
ROOT = os.path.join('..','..')
sys.path.append(os.path.join(CROOT))
from wyrm.data.dictstream import DictStream
from wyrm.data.mltrace import MLTrace
from wyrm.util.time import unix_to_epoch
import wyrm.util.feature_extraction as fex
import wyrm.util.stacking as stk

# PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
EXDF = os.path.join(ROOT, 'example','uw61127051')
EX_URL = 'https://pnsn.org/event/61127051'
PICK = os.path.join(ROOT,'example','AQMS_event_mag_phase_query_output.csv')


pfstr = os.path.join(EXDF,'{model}','{weight}','{site}','{file}')
# Load bulk as well...

mwd = {'EQTransformer': ['stead','pnw','iquique','lendb','instance'],
       'PhaseNet': ['stead','diting','iquique','lendb','instance']}

sites_to_load = {'6Cbs': ['UW.GNW'],
                 '3Csm': ['UW.PCEP'],
                 '3Cbb': ['CC.OBSR'],
                 '1Csp': ['UW.GPW', 'UW.LOC'],
                 '4Css': ['UW.GHW']}
sites = []
for _v in sites_to_load.values():
    sites += _v

# Get picks from AQMS output
evid_list = [int(os.path.split(EXDF)[-1][2:])]
# Index by ARID
pick_df = pandas.read_csv(PICK, index_col='arid')
# Convert pick times from UNIX to UTCDateTime (applys leap corrections)
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x: obspy.UTCDateTime(unix_to_epoch(float(x))))
pick_df = pick_df[pick_df.evid.isin(evid_list)]


dst_tmp = DictStream(obspy.read(os.path.join(EXDF,'bulk.mseed')))
dst = DictStream()
for tr in dst_tmp:
    if tr.site in sites:
        dst.extend(tr.copy())


flist = []
for site in sites:
    flist += glob.glob(pfstr.format(model='*',weight='*',site=site, file='*.*.*.??[PS]*.mseed'))
dst_pre = DictStream.read(flist)

dst.extend(dst_pre.copy())

# Clean up for memory
del dst_pre, dst_tmp



### SEMBLANCE FUNCTION FROM CONGCONG YUAN & YIYU NI
import scipy.ndimage as nd
import numpy as np

def neighbor_sum(arr):
    return sum(arr)

def ensemble_semblance(signals, paras):
    '''
    Function: calculate coherence or continuity via semblance analysis.
    Reference: Marfurt et al. 1998
    
    PARAMETERS:
    ---------------
    signals: input data [ntraces, npts]
    paras: a dict contains various parameters used for calculating semblance. 
           Note: see details in Tutorials.
    
    RETURNS:
    ---------------
    semblance: derived cohence vector [npts,]
    
    Written by Congcong Yuan (Jan 05, 2023)
    
    '''
    # setup parameters
    ntr, npts = signals.shape 
    dt = paras['dt']
    semblance_order = paras['semblance_order']
    semblance_win = paras['semblance_win']
    weight_flag = paras['weight_flag']
    window_flag = paras['window_flag']
    
    semblance_nsmps = int(semblance_win/dt)
    
    # initializing
    semblance, v = np.zeros(npts), np.zeros(npts)
    
    # sums over traces
    square_sums = np.sum(signals, axis=0)**2
    sum_squares = np.sum(signals**2, axis=0)
    
    # loop over all time points
    if weight_flag:
        if weight_flag == 'max':
            v = np.amax(signals, axis=0)
        elif weight_flag == 'mean':
            v = np.mean(signals, axis=0)
        elif weight_flag == 'mean_std':
            v_mean = np.mean(signals, axis=0)
            v_std = np.mean(signals, axis=0)
            v = v_mean/v_std
    else:
        v = 1.
        
    if window_flag:
        # sum over time window
        sums_num = nd.generic_filter(square_sums, neighbor_sum, semblance_nsmps, mode='constant')
        sums_den = ntr*nd.generic_filter(sum_squares, neighbor_sum, semblance_nsmps, mode='constant')
    else:
        sums_num = square_sums
        sums_den = ntr*sum_squares

    # original semblance
    semblance0 = sums_num/sums_den

    # enhanced semblance
    semblance = semblance0**semblance_order*v 
    
    return semblance  
#### END OF STRAIGHT COPY OF ELEP SEMBLANCE

sites = [sites[0]]
# Split by site
output_holder = {}
# for site, sdst in dst.split_on_key(key='site').items():
for site in sites:
    sdst = dst.fnselect(f'{site}.*')
    for comp, csdst in sdst.split_on_key(key='component').items():
        # Safety check that we're only including P or S prediction traces
        if comp in ['P','S']:
            pass
        else:
            continue
        output_holder.update({site: {comp:{}}})

        # Get list of models in this component-site subset
        # NOTE: need to include * to have this map as a wildcard call with .isin()
        modset = list(csdst.traces.keys())
        # Create the weight/model powerset
        mod_powerset = stk.powerset(modset, with_null=False)

        for _i, iset in enumerate(mod_powerset):
            print(f'processing set {iset}')
            # Safety catch to not process on the null-set
            if len(iset) == 0:
                continue
            else:
                pass
            # Create a trimmed copy
            icsdst = csdst.isin(iset).copy().trim(starttime=csdst.stats.max_starttime,
                                                    endtime=csdst.stats.min_endtime)
            # Create prediction stack and fold stack
            pstack = np.array([tr.data for tr in icsdst])
            fstack = np.array([tr.fold for tr in icsdst])
            # Run ensemble semblance from Yuan et al. (2023)
            ct = ensemble_semblance(pstack, {'semblance_order': 2,
                                    'semblance_win': 0.5,
                                    'weight_flag': 'max',
                                    'window_flag': True,
                                    'dt': 0.01})
            # Trigger on ensemble semblance trace
            # for thr in threhold_vals:
            #     triggers = 
            output_holder[site][comp].update({_i: {'ct': ct, 'cset': iset}})
    







# def process_det(row, mlt):
#     line = []
#     # Get maximum value of the data
#     Pmax = np.nanmax(mlt.data)
#     # Get the p05 quantile
#     Pp05 = np.quantile(mlt.data, q=0.05)
#     # Get the index of the pick time 
#     ii = mlt.utcdatetime_to_nearest_index(row.arrdatetime)
#     # Get the probability value at the pick time
#     Pv = mlt.data[ii]
#     # Get the fold at the pick time
#     Pf = mlt.fold[ii]
#     line = [mlt.id, arid, row.arrdatetime, Pmax, Pp05, ii, Pv, Pf]
#     # run triggering on a series of probability thresholds
#     for phat in [0.05, 0.1, 0.3, 0.5, 0.8, 0.9]:
#         qtriggers = obspy.signal.trigger.trigger_onset(mlt.data, phat, phat)
#         if len(qtriggers) == 0:
#             has_pick = False
#         else:
#             for qt in qtriggers:
#                 if qt[0] <= ii <= qt[1]:
#                     has_pick = True
#                 else:
#                     has_pick = False
#         line.append(len(qtriggers))
#         line.append(has_pick)
#     evid_ser = pandas.Series(line, index=['pred_id','arid','arrdatetime','Pmax','Pp05','ii','Pv','Pf',
#                                           'Pt0.05_nt','Pt0.05_hp','Pt0.10_nt','Pt0.10_hp','Pt0.30_nt','Pt0.30_hp',
#                                           'Pt0.50_nt','Pt0.50_hp','Pt0.80_nt','Pt0.80_hp','Pt0.90_nt','Pt0.90_hp'])
#     evid_ser.name = evid_ser.pred_id
#     return evid_ser

# # Iterate across EVIDs
# for evid in evid_list:
#     print(f'processing evid uw{evid}')
#     # Iterate across analyst picks from AQMS for uw{EVID}
#     for arid, row in pick_df[pick_df.evid == evid].iterrows():
#         # Get list of predictions for this site/phase combination
#         pred_files = glob.glob(pfstr.format(evid=evid,
#                                            model='*',
#                                            weight='*',
#                                            site=f'{row.net}.{row.sta}',
#                                            file=f'*.*.*.??{row.iphase}.*'))
#         mltlist = [MLTrace.read(pf) for pf in pred_files]
#         dst = DictStream(traces = mltlist)
#         dst.trim(starttime=dst.stats.max_starttime, endtime=dst.stats.min_endtime)
#         pstack = np.c_[[mlt.data for mlt in dst]]
#         fstack = np.c_[[mlt.fold for mlt in dst]]
#         pick_holder = []
#         # semb_full = semblance(pstack, fstack, **sembkwargs)
#         # pick_holder.append(semb_full)
#         # Generate powerset of prediction IDs
#         powset = powerset(range(len(dst)))
#         for iset in powset:
#             bkey = make_bkey(iset, len(dst))
#             ipstack = np.c_[[pstack[_i] for _i in iset]]
#             ifstack = np.c_[[fstack[_i] for _i in iset]]
#             breakpoint()
#             # semb_set = semblance(pstack, fstack, **sembkwargs)

            

            
#     #     for iset in powset:

#     #     for _id, mlt in dst.traces.items():
#     #         # run individual assessment
#     #         result = process_det(row, mlt):
#     #         # run LOOCV assessment

#     #         # run 
#     #     # Run loocv
#     #     for _id in dst.traces.keys():
#     #         semb_mlt = semblance()
#     # breakpoint()
                

            

#         # for tr in dst:

#     # pred_file_list = glob.glob(fstr.format(evid=evid, model='*', weight='*', site='*'))
