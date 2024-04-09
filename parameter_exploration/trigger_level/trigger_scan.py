"""

define pick_offset_* = pick_sample - nearest_trigger_based_sample
        Thus, a pick_sample earlier than the nearest trigger_edge has a negative pick_offset_edge

        ---P---TTTTTMTTT-----
    Edge   |-4 |
    Peak   |  -10   |
        -------TTTTTMTTT---P-
    Edge               | 4 |
    Peak            |   7  |
"""

import sys, os, pandas, glob, time
import numpy as np
ROOT = os.path.join('..','..')
sys.path.append(ROOT)
from wyrm.data.dictstream import DictStream
from wyrm.util.time import unix_to_UTCDateTime
from obspy.signal.trigger import trigger_onset

PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PICK = os.path.join(PROOT,'data','AQMS','AQMS_event_mag_phase_query_output.csv')
PRED = os.path.join(PROOT,'processed_data')

pfstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','{model}','{weight}','{site}','{file}')
sfstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','initial_trigger_assessment_v2.csv')


evid_list = glob.glob(os.path.join(PRED, 'avg_pred_stacks','uw*'))
evid_list.sort()
evid_list = [int(os.path.split(evid)[-1][2:]) for evid in evid_list]
pick_df = pandas.read_csv(PICK, index_col='arid')
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x: unix_to_UTCDateTime(float(x)))
pick_df = pick_df[pick_df.evid.isin(evid_list)]


prob_thresholds = [0.05, 0.1, 0.16,0.2, 0.25,0.3, 0.4, 0.5,0.6,0.7, 0.75,0.8, 0.84, 0.9, 0.95]
trig_length = [5, 9e99]
print('starting iterations')
tick = time.time()
# Iterate across events
for _i, evid in enumerate(evid_list):
    print(f'running {evid} | {_i+1:02}/{len(evid_list)} | elapsed time {time.time() - tick:.1f} sec')
    holder = []
    # Isolate sites from path strings    
    site_paths = glob.glob(os.path.join(PRED,'avg_pred_stacks',f'uw{evid}','*','*','*'))
    sites_tmp = [os.path.split(sp)[-1] for sp in site_paths]
    sites = []
    # Get unique sites for this event
    for site in sites_tmp:
        if site not in sites:
            sites.append(site)
    # Iterate across sites
    for site in sites:
        pfiles = glob.glob(pfstr.format(
            evid=evid,
            model='*',
            weight='*',
            site=site,
            file='*.*.*.??[PS]*'))
        net, sta = site.split('.')
        # LOAD WAVEFORM DATA
        dst = DictStream.read(pfiles)
        # Iterate across labels
        for label, ldst in dst.split_on_key(key='component').items():
            # Subset analyst picks
            pick_idf = pick_df[(pick_df.evid==evid)&\
                               (pick_df.iphase==label)&\
                               (pick_df.sta == sta)&\
                               (pick_df.net == net)]
            # Iterate across mltraces in dictstream
            for tr in ldst:
                t0 = tr.stats.starttime
                sr = tr.stats.sampling_rate
                # Iterate across triggering thresholds
                for pt in prob_thresholds:
                    # RUN TRIGGER #
                    triggers = trigger_onset(tr.data, pt, pt)
                    passing_triggers = []
                    ntrig = len(triggers)
                    for trigger in triggers:
                        tl = trigger[1]- trigger[0]
                        if min(trig_length) <= tl <= max(trig_length):
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
                    if len(pick_idf) == 0:
                        arid = None
                        pick_in_trigger=False
                        nearest_peak_dsamp = None
                        nearest_peak_P = None
                        nearest_peak_width = None
                        pick_Pval = None
                    elif len(pick_idf) == 1:
                        arid = pick_idf.index[0]
                        pick_time = pick_idf.iloc[0].arrdatetime
                        pick_samp = round((pick_time - t0)*sr)
                        try:
                            pick_Pval = tr.data[pick_samp]
                        except IndexError:
                            pick_Pval = None
                        
                        # Values overwritten if there are triggers and picks
                        nearest_peak_dsamp = None
                        nearest_peak_P = None
                        nearest_peak_width = None
                    # Safety catch clause for debugging
                    elif len(pick_idf) > 1:
                        print('somehow got multiple picks...')
                        breakpoint()

                    # If there are triggers and picks
                    if nptrig > 0 and len(pick_idf) == 1 and pick_Pval is not None:
                        # initialize distances as length of trace (overwrites prior None values)
                        nearest_edge_dsamp = tr.stats.npts
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
                    line = [evid, arid, 
                            net, sta, tr.stats.location, tr.stats.channel,
                            tr.stats.component, tr.stats.model, tr.stats.weight,
                            pt, ntrig, nptrig,
                            pick_in_trigger,pick_Pval,
                            nearest_peak_dsamp, nearest_peak_P, nearest_peak_width]
                    holder.append(line)

    df_evid_result = pandas.DataFrame(holder, columns=['evid','arid',
                                                       'net','sta','loc','chan',
                                                       'comp','model','weight',
                                                       'thresh','raw_trigger_ct','trigger_ct',
                                                       'pick_in_trigger','pick_Pval',
                                                       'nearest_peak_dsmp','nearest_peak_Pval','nearest_peak_width'])
    df_evid_result.to_csv(sfstr.format(evid=evid), header=True, index=False)
             



