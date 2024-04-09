import sys, os, pandas, glob, time
ROOT = os.path.join('..','..')
sys.path.append(ROOT)
from wyrm.data.dictstream import DictStream
from wyrm.util.time import unix_to_UTCDateTime
from obspy.signal.trigger import trigger_onset

PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PICK = os.path.join(PROOT,'data','AQMS','AQMS_event_mag_phase_query_output.csv')
PRED = os.path.join(PROOT,'processed_data')

pfstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','{model}','{weight}','{site}','{file}')
sfstr = os.path.join(PRED,'avg_pred_stacks','uw{evid}','initial_trigger_assessment.csv')


evid_list = glob.glob(os.path.join(PRED, 'avg_pred_stacks','uw*'))
evid_list.sort()
evid_list = [int(os.path.split(evid)[-1][2:]) for evid in evid_list]
pick_df = pandas.read_csv(PICK, index_col='arid')
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x: unix_to_UTCDateTime(float(x)))
pick_df = pick_df[pick_df.evid.isin(evid_list)]


prob_thresholds = [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95]
print('starting iterations')
tick = time.time()
for _i, evid in enumerate(evid_list[2:]):
    print(f'running {evid} | {_i+1:02}/{len(evid_list)} | elapsed time {time.time() - tick:.1f} sec')
    holder = []
    
    # Load all event predictions
    site_paths = glob.glob(os.path.join(PRED,'avg_pred_stacks',f'uw{evid}','*','*','*'))
    sites_tmp = [os.path.split(sp)[-1] for sp in site_paths]
    sites = []
    for site in sites_tmp:
        if site not in sites:
            sites.append(site)
    for site in sites:
        pfiles = glob.glob(pfstr.format(
            evid=evid,
            model='*',
            weight='*',
            site=site,
            file='*.*.*.??[PS]*'))
        net, sta = site.split('.')
        # print(f'loading predictions for uw{evid} - {net}.{sta}')
        dst = DictStream.read(pfiles)
        for label, ldst in dst.split_on_key(key='component').items():
            pick_idf = pick_df[(pick_df.evid==evid)&\
                               (pick_df.iphase==label)&\
                               (pick_df.sta == sta)&\
                               (pick_df.net == net)]
            for tr in ldst:
                t0 = tr.stats.starttime
                sr = tr.stats.sampling_rate
                for pt in prob_thresholds:
                    triggers = trigger_onset(tr.data, pt, pt)
                    ntrig = len(triggers)
                    has_pick = None
                    if ntrig == 0:
                        has_pick = False
                    if len(pick_idf) == 0:
                        arid = None
                        has_pick=False
                    
                    if ntrig > 0 and len(pick_idf) > 0:
                        for arid, row in pick_idf.iterrows():
                            for trigger in triggers:
                                pick_samp = (row.arrdatetime - t0)*sr
                                if trigger[0]<= pick_samp <= trigger[1]:
                                    has_pick=True
                            if has_pick is None:
                                has_pick = False
                    line = [evid, arid, net, sta,\
                            tr.stats.location, tr.stats.channel, tr.stats.component,\
                            tr.stats.model, tr.stats.weight, pt, ntrig, has_pick]
                    holder.append(line)
    df_evid_result = pandas.DataFrame(holder, columns=['evid','arid',
                                                       'net','sta','loc','chan',
                                                       'comp','model','weight',
                                                       'thresh','trigger_ct','has_pick'])
    df_evid_result.to_csv(sfstr.format(evid=evid), header=True, index=False)
             


                       
    # models = glob.glob(os.path.join(PRED,'avg_pred_stacks',f'uw{evid}'))
    # for model in models:

    #     weights = glob.glob(os.path.join(model,'*'))
    # for label in ['P','S']:
    #     pflist = glob.glob(pfstr.format(evid=evid, model='*', weight='*', site='*'))