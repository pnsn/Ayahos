import sys, os, pandas, glob, fnmatch, time
import multiprocessing as mp
import numpy as np
ROOT = os.path.join('..')
sys.path.append(ROOT)
from wyrm.data.dictstream import DictStream
from wyrm.util.time import unix_to_UTCDateTime
from wyrm.util.semblance import powerset
from obspy.signal.trigger import trigger_onset


# def assign_confusion_matrix_line(
#         arid,
#         trigger_ct,
#         pick_in_trigger,
#         nearest_peak_dsamp,
#         correctly_labeled,
#         pick_tol_samp=100):
#     """
#     Based on extracted features from the trigger_scan.py script, assign
#     values for the confusion matrix

#         | TP | FN |
#         | FP | TN |
#     TP = true positive, value 1 assigned if
#         A) an analyst pick is contained within a trigger
#         B) an analyst pick falls within pick_tol_samp samples of a trigger's peak
#          NOTE: analyst picks on one seismic sensor are considered shared with co-located sensors
#             e.g., a pick on UW.GNW.--.BH? is also considered a pick on UW.GNW.--.EN?
#     TN = true negative, value 1 assigned if:
#         A) on data without an associated analyst pick, no triggers occur
#         B) on data with an analyst pick, only one trigger occurs and conforms to the requirements of TP above
#     FP = false positive, value assigned as:
#         A) FP = T triggers for data with an associated pick where all triggers fail to meet
#             requirements of TP above
#         B) FP = T - 1 as in the case above, but one trigger conforms to TP requirements
#         C) FP = T triggers for data without an associated pick
#     FN = false negative, value 1 assigned if data have an associated pick but no triggers satisfy
#         the requirements of TP above.
    
#     :: INPUTS ::
#     :param row: [pandas.Series] a series representation of a row from the output CSV files
#                 produced by trigger_scan.py
#     :param pick_tol_samp: [int] pick tolerance samples to allowed for TP, definition B above.

#     :: OUTPUTS ::
#     :return TP: [int] number of true positive values
#     :return FP: [int] number of false positive values
#     :return TN: [int] number of true negative values
#     :return FN: [int] number of false negative values
#     :return XP: [int] number of incorrectly labeled picks ()
#     :return tol_needed: [bool] - flag if the tolerance bound inclusion clause was used for TP
#     :return labeled_data: [bool] - flag if the C matrix values were determined for labeled or unlabeled data
#     """
#     # If labeled data
#     if arid is None:
#         labeled_data = False
#         tol_needed = None
#         if trigger_ct == 0:
#             TN = 1
#             FP = 0
#             TP = 0
#             FN = 0
#             XP = 0
#         else:
#             FP = trigger_ct
#             TN = 0
#             TP = 0
#             FN = 0
#             XP = 0
#     elif np.isfinite(arid):
#         labeled_data=True
#         # If no triggers
#         if trigger_ct == 0:
#             TP = 0
#             FN = 1
#             FP = 0
#             TN = 0
#             XP = 0
#             tol_needed = None
#         # If there are triggers
#         if trigger_ct > 0:
#             # Trigger includes pick time - TODO: how to handle long triggers???
#             if pick_in_trigger:
#                 if correctly_labeled:
#                     TP = 1
#                     XP = 0
#                     FN = 0
#                     tol_needed = False
#                 else:
#                     TP = 0
#                     XP = 1
#                     FN = 0
#                     tol_needed = False
#             # Trigger meets pick when including tolerance
#             elif abs(nearest_peak_dsamp) <= pick_tol_samp:
#                 if correctly_labeled:
#                     TP = 1
#                     FN = 0
#                     XP = 0
#                     tol_needed = True
#                 else:
#                     TP = 0
#                     XP = 1
#                     FN = 0
#                     tol_needed = True
#             # Triggers exist, but do not meet pick/tolerance
#             else:
#                 TP = 0
#                 FN = 1
#                 XP = 0
#                 tol_needed = None
#             # Assign number of false positives
#             if correctly_labeled:
#                 FP = trigger_ct - TP
#             else:
#                 FP = trigger_ct - XP
#             # If no false positives, grant this one true negative score
#             if FP == 0:
#                 TN = 1
#             else:
#                 TN = 0
#     # If no labeled data are present
#     else:
#         labeled_data = False
#         tol_needed = None
#         if trigger_ct == 0:
#             TN = 1
#             FP = 0
#             TP = 0
#             FN = 0
#             XP = 0
#         else:
#             FP = trigger_ct
#             TN = 0
#             TP = 0
#             FN = 0
#             XP = 0
#     return TP, FP, TN, FN, XP, tol_needed, labeled_data

# # def compose_poolstarmap_iterable(evid_list, pick_df, glob_format_string='uw{evid}/*/*/*'):
# #     iterable = []
# #     for evid in evid_list:
# #         sgs = os.path.join(BASE_PATH, os.path.join(*glob_format_string.split('/')))
# #         site_paths = glob.glob(sgs.format(BASE_PATH=BASE_PATH, evid=evid))
# #         sites = [os.path.split(site_path)[-1] for site_path in site_paths]
# #         picks = pick_df[pick_df.evid==evid]
# #         entry = (evid, picks, sites)
# #         iterable.append(entry)
# #     return iterable



## CORE PROCESS
def semblance_trigger_site_event(evid, site):
    time.sleep(0.05)
    read_ele = [BASE_PATH] + READ_FSTR.split('/')
    save_ele = [BASE_PATH] + SAVE_FSTR.split('/')
    read_fstring = os.path.join(*read_ele)
    save_fstring = os.path.join(*save_ele)
    # Get file list for all evid-site-model-weight bounded prediction traces
    pfiles = []
    for model, weight in MODSET:
        glob_str = read_fstring.format(evid=evid,model=model,weight=weight,site=site)
        pfiles += glob.glob(glob_str)
    
    # Create holder for site-level triggering ouput capture
    holder = {}

    # Extract net and sta strings from the site name
    net, sta = site.split('.')

    # # Iterate across predicted labels
    # for label in ['P','S']:
    # Filter to find file names that meet the site-label requirements
    matches = fnmatch.filter(pfiles, f'*/{site}*.mseed')
    #### LOAD PREDICTION DATA FOR SPECIFIC EVENT SITE AND LABEL ####
    dst = DictStream.read(matches)
    if len(dst) == 0:
        print(f'xx-- skipping {evid} {site} ---- no data')
        return pandas.DataFrame()
    # Get picks for this site (apply to multiple instruments, if applicable)
    pick_idf = PICKS[(PICKS.evid==evid)&(PICKS.sta == sta)&(PICKS.net == net)]
    # Split predictions on instrument
    # Create an internal holder for semblance traces (dev comparison tool)
    semb_dst = DictStream()
    for instrument, idst in dst.split_on_key(key='instrument').items():
        # Subset again to match this site, event, label
        output_name = save_fstring.format(evid=evid, instrument=instrument)
        if os.path.isfile(output_name):
            print(f'ss-- skipping {evid} {instrument} | already exists ----')
            continue

        for comp, sidst in idst.split_on_key(key='component').items():
            # Create powerset seed for a given event-instrument-label combination
            powerset_seed = list(sidst.traces.keys())
            # Create id powerset
            id_powerset = powerset(powerset_seed, with_null=False)
            # Conduct semblance for each - NOTE: DictStream.semblance() does single-trace passthrough
            for iset in id_powerset:
                # Get subset of prediction traces that match this model set
                sldst = dst.isin(iset)
                # RUN SEMBLANCE
                mlt_semb = process_semblance(sldst)
                # Safety catch from debugging (probably obsolite)
                if mlt_semb.id == '..--...':
                    breakpoint()
                # Safety catch for null_set input
                if mlt_semb is None:
                    continue
                else:
                    semb_dst.extend(mlt_semb)
                # Conduct triggering
                for pt in THRESHOLDS:
                    outputs = process_triggering(mlt_semb, pt, pick_idf, evid)
                    for output in outputs:
                        for _k, _v in output.items():
                            if _k not in holder.keys():
                                holder.update({_k:[]})
                            holder[_k].append(_v)
                    # for line in trig_stats:
                    #     holder.append(line)
        semb_dst
        # Save for each instrument
        df_out = pandas.DataFrame(holder)
        # output_name = save_fstring.format(evid=evid,site=site)
        # breakpoint()

        print(f'---- writing {evid} {instrument} to disk ----')
        df_out.to_csv(output_name, header=True, index=False)

    return df_out

def process_semblance(dst):
    ## CONDUCT SEMBLANCE ##
    if len(dst) > 1:
        tr = dst.semblance(**PARAS)
        #     window_len=1,
        #     order=2,
        #     coefficient='max',
        #     fold_weighted=False)
    ## IF SET OF 1 PASS SINGLE TRACE ##
    elif len(dst) == 1:
        tr = dst[0]
    else:
        return None
    return tr

                         
def process_triggering(tr, threshold, pick_df, evid, trigger_limits=[5,9e99], pick_tol_samp=100):
    outputs = []
    
    columns=['evid','arid','iphase',
             'network','station','location','channel',
             'component','model','weight',
             'thresh','raw_trigger_ct','trigger_ct',
             'pick_in_trigger','pick_Pval',
             'nearest_peak_dsamp','nearest_peak_Pval','nearest_peak_width',
             'TP','FP','TN','FN','XP','tol_needed','labeled_data','correctly_labeled']
    out_dict = dict(zip(columns, [None]*len(columns)))
    out_dict.update({'evid': evid,'thresh': threshold})
    for _k in ['network','station','location','channel','model','weight']:
        out_dict.update({_k: tr.stats[_k]})
    out_dict.update({'component': tr.comp})

    # Run triggering
    triggers = trigger_onset(tr.data, threshold, threshold)
    out_dict.update({'raw_trigger_ct': len(triggers)})
    # Filter triggers to remove too short and too long
    passing_triggers = []
    for trigger in triggers:
        tl = trigger[1]- trigger[0]
        if min(trigger_limits) <= tl <= max(trigger_limits):
            passing_triggers.append(trigger)
    passing_triggers = np.array(passing_triggers, dtype=np.int64)
    out_dict.update({'trigger_ct': len(passing_triggers)})

    if out_dict['trigger_ct'] == 0 or len(pick_df) == 0:
        out_dict.update({'pick_in_trigger': False})

    # Process if there are picks 
    if len(pick_df) > 0:
        # Iterate across picks
        for arid, row in pick_df.iterrows():
            iout_dict = out_dict.copy()
            iout_dict.update({'iphase': row.iphase, 'arid': arid})


            if iout_dict['iphase'] == tr.comp:
                iout_dict.update({'correctly_labeled':True})
            else:
                iout_dict.update({'correctly_labeled':False})

            pick_time = row.arrdatetime
            pick_samp = round((pick_time - tr.stats.starttime)*tr.stats.sampling_rate)
            try:
                iout_dict.update({'pick_Pval': tr.data[pick_samp]})
            except IndexError:
                pass
            # If there are triggers to process and a pick inside the prediction timeseries
            if len(passing_triggers) > 0 and tr.data[pick_samp] is not None:
                # nearest_edge_dsamp = tr.stats.npts
                nearest_peak_dsamp = tr.stats.npts
                # Scan across all triggers
                for trigger in passing_triggers:
                    # Get sample location of the peak of this trigger
                    max_p_samp = np.nanargmax(tr.data[trigger[0]:trigger[1]])
                    # Get maximum probability value of trigger
                    max_p_val = np.nanmax(tr.data[trigger[0]:trigger[1]])
                    # Get sample distance from pick to peak
                    inpdn = pick_samp - trigger[0] + max_p_samp
                    if trigger[0] <= pick_samp <= trigger[1]:
                        iout_dict.update({'pick_in_trigger': True,
                                        'nearest_peak_dsamp': inpdn,
                                        'nearest_peak_Pval': max_p_val,
                                        'nearest_peak_width': trigger[1] - trigger[0]})
                        break
                    # otherwise, if pick_sample is not inside this trigger
                    else:
                        # See if the pick-peak distance magnitude is the smallest yet
                        if np.abs(inpdn) < np.abs(nearest_peak_dsamp):
                            iout_dict.update({'pick_in_trigger': False,
                                               'nearest_peak_dsamp': inpdn,
                                               'nearest_peak_Pval': max_p_val,
                                               'nearest_peak_width': trigger[1] - trigger[0]})
            # Run confusion matrix processing at the pick level
            if iout_dict['trigger_ct'] == 0:
                iout_dict.update({'TN': 0, 'TP': 0, 'FN': 1, 'FP': 0, 'XP': 0})
            else:
                if iout_dict['pick_in_trigger']:
                    iout_dict.update({'tol_needed': False})
                    if iout_dict['correctly_labeled']:
                        iout_dict.update({'TP': 1, 'XP': 0, 'FN': 0})
                    else:
                        iout_dict.update({'TP': 0, 'XP': 1, 'FN': 0})
                elif abs(iout_dict['nearest_peak_dsamp']) <= pick_tol_samp:
                    iout_dict.update({'tol_needed': True})
                    if iout_dict['correctly_labeled']:
                        iout_dict.update({'TP': 1, 'FN': 0, 'XP': 0})
                    else:
                        iout_dict.update({'TP': 0, 'FN': 0, 'XP': 1})
                else:
                    iout_dict.update({'TP': 0, 'FN': 1, 'XP': 0})
                if iout_dict['correctly_labeled']:
                    iout_dict.update({'FP': iout_dict['trigger_ct'] - iout_dict['TP']})
                else:
                    iout_dict.update({'FP': iout_dict['trigger_ct'] - iout_dict['XP']})
                if iout_dict['FP'] == 0:
                    iout_dict.update({'TN': 1})
                else:
                    iout_dict.update({'TN': 0})
            outputs.append(iout_dict)

    # If there were no picks to process
    else:
        out_dict.update({'TP': 0, 'FN': 0, 'XP': 0})
        if out_dict['trigger_ct'] == 0:
            out_dict.update({'TN': 1, 'FP': 0})
        else:
            out_dict.update({'TN': 0, 'FP': out_dict['trigger_ct']})
        outputs.append(out_dict)

    
    return outputs



    



## LOADING HELPER FUNCTIONS
def load_picks(pick_csv, evids=None, **options):
    # Load full data file
    pick_df = pandas.read_csv(pick_csv, **options)
    # Subset if an input is provided
    if evids is not None:
        pick_df = pick_df[pick_df.evid.isin(evids)]
    # Change arrival and origin times to obspy.UTCDateTime
    pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x: unix_to_UTCDateTime(float(x)))
    pick_df.datetime = pick_df.datetime.apply(lambda x: unix_to_UTCDateTime(float(x)))
    return pick_df

def get_sitelist(evid_dir):
    glob_str = os.path.join(evid_dir,'*','*','*')
    pfiles = glob.glob(glob_str)
    sites = []
    for pf in pfiles:
        isite = os.path.split(pf)[-1]
        if isite not in sites:
            sites.append(isite)
    return sites

def init_worker(paras, pick_df, thresholds, modset, trig_limits, paths):
    global PARAS, PICKS, THRESHOLDS, MODSET, BASE_PATH, READ_FSTR, TRIG_LIMITS, SAVE_FSTR
    PARAS = paras
    PICKS = pick_df
    THRESHOLDS = thresholds
    MODSET = modset
    BASE_PATH = paths['base_path']
    READ_FSTR = paths['read_fstr']
    SAVE_FSTR = paths['save_fstr']
    TRIG_LIMITS = trig_limits

#### IF MAIN ####
if __name__ == '__main__':
    run_parallel = True
    n_pool = 10
    n_chunk = 1000

    # Assign global path variables
    # Get path to picks file
    PROOT = '/Users/nates/Documents/Conferences/2024_SSA/PNSN'
    # Pick file path/name
    pick_file = os.path.join(PROOT,'data','AQMS','AQMS_event_mag_phase_query_output.csv')

    # BASE_PATH used in internal functions
    base_path = os.path.join(PROOT,'processed_data','avg_pred_stacks')
    # READ_/SAVE_STRUCTURE used in internal functions
    read_structure='uw{evid}/{model}/{weight}/{site}/*.*.*.??[PS].*'
    save_structure='uw{evid}/{instrument}__semb_trigger_scan.csv'

    paths = {'base_path': base_path,
             'read_fstr': read_structure,
             'save_fstr': save_structure}

    # MODSET for powerset semblance checks
    modset = [('EQTransformer','pnw'), ('EQTransformer','instance'),
              ('EQTransformer', 'stead'), ('PhaseNet','diting')]
    if len(modset) > 5: 
        print(f'WARNING: this modset will produce {2**len(modset) -1} to {2**(2*len(modset)) - 1} semblance traces per site!')
        breakpoint()

    # Set probability thresholds for triggering experiments
    # NOTE: Semblance trigger threshold in ELEP is set much lower (0.05) compared to 
    #       individual model thresholds in the literature (\tau \in [0.2, 0.5])
    thresholds = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    # Set min/max lengths for acceptable triggers [samples]
    trig_limits = [5, 9e99]
    # paras
    paras = {'window_len': 0.5, 'order': 2, 'coefficient': 'max',
             'fold_weighted': False, 'fill_value': 0, 'trim_type': 'inner'}
    # GET EVID SUBSET
    evid_dir_list = glob.glob(os.path.join(base_path,'uw*'))
    evid_list = [int(os.path.split(evid_dir)[-1][2:]) for evid_dir in evid_dir_list]
    # GET PICKS FROM FILE
    pick_df = load_picks(pick_file, evids=evid_list, index_col='arid')

    evid_site_sets = []
    for evid, evid_dir in zip(evid_list, evid_dir_list):
        sites = get_sitelist(evid_dir)
        # instruments = get_instrument_list(evid_dir)
        sites.sort()
        for site in sites:
            estup = (evid, site)
            evid_site_sets.append(estup)
    
    picked_evid_site_sets = [tuple([_e[0], f'{_e[1]}.{_e[2]}']) for _e in pick_df[['evid','net','sta']].value_counts().index]

    job_input = picked_evid_site_sets

    # breakpoint()
    initargs = (paras, pick_df, thresholds, modset, trig_limits, paths)
    if run_parallel:
        # breakpoint()
        tick = time.time()
        print(f'==== LAUNCHING POOL FOR {len(job_input)} JOBS BY {n_pool} WORKERS ====')
        with mp.Pool(n_pool, initializer=init_worker, initargs=initargs) as p:
            sout = p.starmap(semblance_trigger_site_event, job_input, chunksize=n_chunk)
        tock = time.time()
        print(f'!DONE! | runtime: {(tock - tick)/60:.2f} min | {(tock - tick)/len(evid_site_sets)} sec/job')
    else:
        init_worker(*initargs)
        for evid, site in job_input:
            df_evid_result = semblance_trigger_site_event(evid, site)