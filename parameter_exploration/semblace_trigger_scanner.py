import sys, os, pandas, glob, fnmatch, time
import multiprocessing as mp
import numpy as np
ROOT = os.path.join('..')
sys.path.append(ROOT)
from wyrm.data.dictstream import DictStream
from wyrm.util.time import unix_to_UTCDateTime
from wyrm.util.stacking import powerset
from obspy.signal.trigger import trigger_onset

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

def init_worker(pick_df, thresholds, modset, trig_limits, paths):
    global PICKS, THRESHOLDS, MODSET, BASE_PATH, READ_FSTR, TRIG_LIMITS, SAVE_FSTR
    PICKS = pick_df
    THRESHOLDS = thresholds
    MODSET = modset
    BASE_PATH = paths['base_path']
    READ_FSTR = paths['read_fstr']
    SAVE_FSTR = paths['save_fstr']
    TRIG_LIMITS = trig_limits

# def compose_poolstarmap_iterable(evid_list, pick_df, glob_format_string='uw{evid}/*/*/*'):
#     iterable = []
#     for evid in evid_list:
#         sgs = os.path.join(BASE_PATH, os.path.join(*glob_format_string.split('/')))
#         site_paths = glob.glob(sgs.format(BASE_PATH=BASE_PATH, evid=evid))
#         sites = [os.path.split(site_path)[-1] for site_path in site_paths]
#         picks = pick_df[pick_df.evid==evid]
#         entry = (evid, picks, sites)
#         iterable.append(entry)
#     return iterable



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
    holder = []

    # Extract net and sta strings from the site name
    net, sta = site.split('.')

    # Iterate across predicted labels
    for label in ['P','S']:
        # Filter to find file names that meet the site-label requirements
        matches = fnmatch.filter(pfiles, f'*/{site}.*.??{label}.*.mseed')

        #### LOAD WAVEFORM DATA FOR SPECIFIC EVENT SITE AND LABEL ####
        dst = DictStream.read(matches)
        # Subset analyst picks to match this site, event, label
        # breakpoint()
        pick_idf = PICKS[(PICKS.evid==evid)&(PICKS.iphase==label)&(PICKS.sta == sta)&(PICKS.net == net)]
        # Create powerset seed
        powerset_seed = list(dst.traces.keys())
        id_powerset = powerset(powerset_seed, with_null=False)
        for iset in id_powerset:
            sldst = dst.isin(iset)
            mlt_semb = process_semblance(sldst)
            if mlt_semb.id == '..--...':
                breakpoint()
            # print(f'{evid} | {mlt_semb.id}')
            if mlt_semb is None:
                continue
            for pt in THRESHOLDS:
                trig_stats = process_triggering(mlt_semb, pt, pick_idf, evid)
                holder.append(trig_stats)

    # Save for each site
    df_out = pandas.DataFrame(holder, columns=['evid','arid',
                                                'net','sta','loc','chan',
                                                'comp','model','weight',
                                                'thresh','raw_trigger_ct','trigger_ct',
                                                'pick_in_trigger','pick_Pval',
                                                'nearest_peak_dsmp','nearest_peak_Pval','nearest_peak_width',
                                                'TP','FP','TN','FN','tol_needed','labeled_data'])
    output_name = save_fstring.format(evid=evid,site=site)
    print(f'---- writing {evid} {site} to disk ----')
    df_out.to_csv(output_name, header=True, index=False)

    return df_out

def process_semblance(dst):
    ## CONDUCT SEMBLANCE ##
    if len(dst) > 1:
        tr = dst.semblance(
            window_len=1,
            order=2,
            coefficient='max',
            fold_weighted=False)
    ## IF SET OF 1 PASS SINGLE TRACE ##
    elif len(dst) == 1:
        tr = dst[0]
    else:
        return None
    return tr

                         
def process_triggering(tr, pt, pick_df, evid):
    # RUN TRIGGER #
    triggers = trigger_onset(tr.data, pt, pt)
    passing_triggers = []
    ntrig = len(triggers)
    for trigger in triggers:
        tl = trigger[1]- trigger[0]
        if min(TRIG_LIMITS) <= tl <= max(TRIG_LIMITS):
            passing_triggers.append(trigger)
    passing_triggers = np.array(passing_triggers, dtype=np.int64)
    nptrig = len(passing_triggers)
    # Set default value for pick_in_trigger
    pick_in_trigger = None
    # If no triggers, can't have a pick on the analyst pick (if one exists or not)
    if nptrig == 0:
        pick_in_trigger = False
        nearest_peak_dsamp = None
        nearest_peak_P = None
        nearest_peak_width = None
    # If there are no analyst picks to begin with, 
    # flag with arid = None 
    # and by necessity, pick_in_trigger must be False
    if len(pick_df) == 0:
        arid = None
        pick_in_trigger=False
        nearest_peak_dsamp = None
        nearest_peak_P = None
        nearest_peak_width = None
        pick_Pval = None
    elif len(pick_df) == 1:
        arid = pick_df.index[0]
        pick_time = pick_df.iloc[0].arrdatetime
        pick_samp = round((pick_time - tr.stats.starttime)*tr.stats.sampling_rate)
        try:
            pick_Pval = tr.data[pick_samp]
        except IndexError:
            pick_Pval = None
        
        # Values overwritten if there are triggers and picks
        nearest_peak_dsamp = None
        nearest_peak_P = None
        nearest_peak_width = None
    # Safety catch clause for debugging
    elif len(pick_df) > 1:
        print('somehow got multiple picks...')
        breakpoint()

    # If there are triggers and picks
    if nptrig > 0 and len(pick_df) == 1 and pick_Pval is not None:
        # initialize distances as length of trace (overwrites prior None values)
        # nearest_edge_dsamp = tr.stats.npts
        nearest_peak_dsamp = tr.stats.npts
        # Scan across all triggers
        for trigger in passing_triggers:
            # Get maximum probability position in trigger
            max_p_samp = np.nanargmax(tr.data[trigger[0]:trigger[1]])
            # Get maximum probability value of trigger
            max_p_val = np.nanmax(tr.data[trigger[0]:trigger[1]])
            # # Get minimum distance from pick to edge
            # inedn = np.min(np.abs(pick_samp - np.array([trigger])))
            # Get distance from pick to peak
            inpdn = pick_samp - trigger[0] + max_p_samp
            # If pick_sample is inside this trigger
            if trigger[0] <= pick_samp <= trigger[1]:
                # Assign pick_in_trigger flag to True
                pick_in_trigger=True
                # assign delta samples to peak
                nearest_peak_dsamp = inpdn
                # assign peak P value for nearest peak
                nearest_peak_P = max_p_val
                # Stop iterating, we have a match!
                nearest_peak_width = trigger[1] - trigger[0]
                break
            # otherwise, if pick_sample is not inside this trigger
            else:
                # See if the pick-peak distance magnitude is the smallest yet
                if np.abs(inpdn) < np.abs(nearest_peak_dsamp):
                    # If so, update the nearest (signed) distance
                    nearest_peak_dsamp = inpdn
                    # And update the P value for the nearest pick
                    nearest_peak_P = max_p_val
                    nearest_peak_width = trigger[1] - trigger[0]
        # End of trigger loop
    # Final catch, if pick_in_trigger has not been reassigned from default
    # Set pick_in_trigger as False
    if pick_in_trigger is None:
        pick_in_trigger = False

    cstats = assign_confusion_matrix_line(
        arid,
        nptrig,
        pick_in_trigger,
        nearest_peak_dsamp
    ) 

    out = [evid, arid,
           tr.stats.network, tr.stats.station, tr.stats.location,
           tr.stats.channel, tr.comp, tr.stats.model, tr.stats.weight,
           np.round(pt, decimals=2), ntrig, nptrig,
           pick_in_trigger,pick_Pval,
           nearest_peak_dsamp, nearest_peak_P, nearest_peak_width]
    out += cstats
    return out


def assign_confusion_matrix_line(
        arid,
        trigger_ct,
        pick_in_trigger,
        nearest_peak_dsamp,
        pick_tol_samp=100):
    """
    Based on extracted features from the trigger_scan.py script, assign
    values for the confusion matrix

        | TP | FN |
        | FP | TN |
    TP = true positive, value 1 assigned if
        A) an analyst pick is contained within a trigger
        B) an analyst pick falls within pick_tol_samp samples of a trigger's peak
         NOTE: analyst picks on one seismic sensor are considered shared with co-located sensors
            e.g., a pick on UW.GNW.--.BH? is also considered a pick on UW.GNW.--.EN?
    TN = true negative, value 1 assigned if:
        A) on data without an associated analyst pick, no triggers occur
        B) on data with an analyst pick, only one trigger occurs and conforms to the requirements of TP above
    FP = false positive, value assigned as:
        A) FP = T triggers for data with an associated pick where all triggers fail to meet
            requirements of TP above
        B) FP = T - 1 as in the case above, but one trigger conforms to TP requirements
        C) FP = T triggers for data without an associated pick
    FN = false negative, value 1 assigned if data have an associated pick but no triggers satisfy
        the requirements of TP above.
    
    :: INPUTS ::
    :param row: [pandas.Series] a series representation of a row from the output CSV files
                produced by trigger_scan.py
    :param pick_tol_samp: [int] pick tolerance samples to allowed for TP, definition B above.

    :: OUTPUTS ::
    :return TP: [int] number of true positive values
    :return FP: [int] number of false positive values
    :return TN: [int] number of true negative values
    :return FN: [int] number of false negative values
    :return tol_needed: [bool] - flag if the tolerance bound inclusion clause was used for TP
    :return labeled_data: [bool] - flag if the C matrix values were determined for labeled or unlabeled data
    """
    # If labeled data
    if arid is None:
        labeled_data = False
        tol_needed = None
        if trigger_ct == 0:
            TN = 1
            FP = 0
            TP = 0
            FN = 0
        else:
            FP = trigger_ct
            TN = 0
            TP = 0
            FN = 0
    elif np.isfinite(arid):
        labeled_data=True
        # If no triggers
        if trigger_ct == 0:
            TP = 0
            FN = 1
            FP = 0
            TN = 0
            tol_needed = None
        # If there are triggers
        if trigger_ct > 0:
            # Trigger includes pick time - TODO: how to handle long triggers???
            if pick_in_trigger:
                TP = 1
                FN = 0
                tol_needed = False
            # Trigger meets pick when including tolerance
            elif abs(nearest_peak_dsamp) <= pick_tol_samp:
                TP = 1
                FN = 0
                tol_needed = True
            # Triggers exist, but do not meet pick/tolerance
            else:
                TP = 0
                FN = 1
                tol_needed = None
            # Assign number of false positives
            FP = trigger_ct - TP
            # If no false positives, grant this one true negative score
            if FP == 0:
                TN = 1
            else:
                TN = 0
    # If no labeled data are present
    else:
        labeled_data = False
        tol_needed = None
        if trigger_ct == 0:
            TN = 1
            FP = 0
            TP = 0
            FN = 0
        else:
            FP = trigger_ct
            TN = 0
            TP = 0
            FN = 0
    return TP, FP, TN, FN, tol_needed, labeled_data
    
    # df_evid_result.to_csv(sfstr.format(evid=evid), header=True, index=False)







if __name__ == '__main__':
    run_parallel = True
    n_pool = 10
    n_chunk = 100

    # Assign global path variables
    # Get path to picks file
    PROOT = '/Users/nates/Documents/Conferences/2024_SSA/PNSN'
    # Pick file path/name
    pick_file = os.path.join(PROOT,'data','AQMS','AQMS_event_mag_phase_query_output.csv')

    # BASE_PATH used in internal functions
    base_path = os.path.join(PROOT,'processed_data','avg_pred_stacks')
    # READ_/SAVE_STRUCTURE used in internal functions
    read_structure='uw{evid}/{model}/{weight}/{site}/*.*.*.??[PS].*'
    save_structure='uw{evid}/{site}_semb_trigger_scan.csv'

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
    thresholds = [0.05,0.1,0.2,0.3,0.4,0.5]
    # Set min/max lengths for acceptable triggers [samples]
    trig_limits = [5, 9e99]

    # GET EVID SUBSET
    evid_dir_list = glob.glob(os.path.join(base_path,'uw*'))
    evid_list = [int(os.path.split(evid_dir)[-1][2:]) for evid_dir in evid_dir_list][:1]
    # GET PICKS FROM FILE
    pick_df = load_picks(pick_file, evids=evid_list, index_col='arid')

    evid_site_sets = []
    for evid, evid_dir in zip(evid_list, evid_dir_list):
        sites = get_sitelist(evid_dir)
        sites.sort()
        for site in sites:
            estup = (evid, site)
            evid_site_sets.append(estup)
    
    picked_evid_site_sets = []
    for evid, site in evid_site_sets:
        net, sta = site.split('.')
        if len(pick_df[(pick_df.evid==evid)&(pick_df.sta==sta)]) > 0:
            picked_evid_site_sets.append((evid, site))


    initargs = (pick_df, thresholds, modset, trig_limits, paths)
    if run_parallel:
        # breakpoint()
        tick = time.time()
        print('==== LAUNCHING POOLED WORK ====')
        with mp.Pool(n_pool, initializer=init_worker, initargs=initargs) as p:
            sout = p.starmap(semblance_trigger_site_event, picked_evid_site_sets)
        tock = time.time()
        print(f'!DONE! | runtime: {(tock - tick)/60:.2f} min | {(tock - tick)/len(evid_site_sets)} sec/job')
    else:
        init_worker(*initargs)
        for evid, site in picked_evid_site_sets:
            df_evid_result = semblance_trigger_site_event(evid, site)