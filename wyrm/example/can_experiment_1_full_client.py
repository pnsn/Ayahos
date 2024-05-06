import obspy, os, sys, glob, time
import pandas as pd
import numpy as np
sys.path.append(os.path.join('..','..'))
import seisbench.models as sbm
import wyrm.core.wyrmstream as ds
import wyrm.streaming.windowstream as cs
import wyrm.coordinating.coordinate as coor
import wyrm.processing.process as proc 
from wyrm.util.time import unix_to_UTCDateTime


client = obspy.clients.fdsn.Client('IRIS')

common_pt = ['stead','instance','iquique','lendb']
EQT = sbm.EQTransformer()
EQT_list = common_pt + ['pnw']
EQT_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN = sbm.PhaseNet()
PN_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN_list = common_pt + ['diting']

EQT_pick_kwargs = {'thresh': 0.1, 'blinding': (500,500)}
PN_pick_kwargs = {'thresh': 0.1, 'blinding': (250,250)}
## (ADDITIONAL) DATA SAMPLING HYPERPARAMETERS ##
# reference_sampling_rate 
RSR = 100.
#reference_channel_fill_rule
RCFR= 'zeros'
# Reference component
RCOMP = 'Z'

# Initialize WindowWyrm elements
# PARAM NOTE: Set these to 'network' pulses to generate all windows up front
windwyrmEQT = proc.WindowWyrm(
    component_aliases=EQT_aliases,
    model_name='EQTransformer',
    reference_sampling_rate=RSR,
    reference_npts=6000,
    reference_overlap=1800,
    pulse_type='network',
    max_pulse_size=10,
    debug=False)

windwyrmPN = proc.WindowWyrm(
    component_aliases=PN_aliases,
    model_name='PhaseNet',
    reference_sampling_rate=RSR,
    reference_npts=3001,
    reference_overlap=900,
    pulse_type='network',
    max_pulse_size=20,
    debug=False)


# Initialize main pre-processing MethodWyrm objects (these can be cloned for multiple tubes)
# For treating gappy data
mwyrm_gaps = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='treat_gaps',
    pkwargs={})

# For synchronizing temporal sampling and windowing
mwyrm_sync = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='sync_to_reference',
    pkwargs={'fill_value': 0})

# For filling data out to 3-C from non-3-C data (either missing channels, or 1C instruments)
mwyrm_fill = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='apply_fill_rule',
    pkwargs={'rule': RCFR})

# Initialize model specific normalization MethodWyrms
mwyrm_normEQT = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'peak'}
)

mwyrm_normPN = proc.MethodWyrm(
    pclass=cs.ComponentStream,
    pmethod='normalize_traces',
    pkwargs={'norm_type': 'std'}
)

# Initialize PredictionWyrm elements
predwyrmEQT = proc.PredictionWyrm(
    model=EQT,
    weight_names=EQT_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=1000,
    debug=False)

predwyrmPN = proc.PredictionWyrm(
    model=PN,
    weight_names=PN_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=1000,
    debug=False)

# Initialize MethodWyrm pickers
owyrm_pickEQT = proc.OutputWyrm(
    pclass=ds.DictStream,
    pmethod='prediction_trigger_report',
    pkwargs={'thresh': 0.1, 'blinding': (500,500), 'stats_pad': 20,
             'extra_quantiles': [0.025, 0.159, 0.25, 0.75, 0.841, 0.975]})

owyrm_pickPN = proc.OutputWyrm(
    pclass=ds.DictStream,
    pmethod='prediction_trigger_report',
    pkwargs={'thresh': 0.1, 'blinding': (250,250), 'stats_pad': 20,
             'extra_quantiles': [0.025, 0.159, 0.25, 0.75, 0.841, 0.975]})

# # Terminating function
# def concat_save(queue, base_path='.', path_format='{network}.{station}', save_format='pick_report_{isostart}_{isoend}.csv'):
#     df_gather = pd.DataFrame()
#     for _i in range(len(queue)):
#         _x = queue.pop()
#         if isinstance(_x, pd.DataFrame):
#             df_gather = pd.concat([df_gather, _x], axis=0, ignore_index=True)
#         elif _x is None:
#             continue
#         else:
#             queue.appendleft(_x)

        



# # Initialize Prediction BufferWyrm elements
# pbuffwyrmEQT = coor.BufferWyrm(max_length=200,
#                               restrict_past_append=True,
#                               blinding=(500,500),
#                               method=3,
#                               max_pulse_size=10000,
#                               debug=False)

# pbuffwyrmPN = coor.BufferWyrm(max_length=200,
#                              restrict_past_append=True,
#                              blinding=(250,250),
#                              method=3,
#                              max_pulse_size=10000,
#                              debug=False)


## GROUP INDIVIDUAL WYRMS
# Compose EQT processing TubeWyrm
tubewyrmEQT = coor.TubeWyrm(
    max_pulse_size=1,
    debug=True,
    wyrm_dict= {'window': windwyrmEQT,
                'gaps': mwyrm_gaps.copy(),
                'sync': mwyrm_sync.copy(),
                'norm': mwyrm_normEQT,
                'fill': mwyrm_fill.copy(),
                'predict': predwyrmEQT,
                'pick': owyrm_pickEQT})
    
# Copy/Update to create PhaseNet processing TubeWyrm
tubewyrmPN = tubewyrmEQT.copy().update({'window': windwyrmPN,
                                        'norm': mwyrm_normPN,
                                        'predict': predwyrmPN,
                                        'pick': owyrm_pickPN})

## GROUP TUBES
canwyrm = coor.CanWyrm(wyrm_dict={'EQTransformer': tubewyrmEQT,
                                  'PhaseNet': tubewyrmPN},
                       wait_sec=0,
                       max_pulse_size=30,
                       debug=False)

print('Wyrm Module Compiled')

ROOT = os.path.join('..','..')
# Define root directory path
HDD_ROOT = os.path.join('/Volumes','TheWall','PNSN_miniDB','data')
# REPORT_FILE = os.path.join(HDD_ROOT,'processing_log_pt_2.txt')
# if os.path.exists(REPORT_FILE):
#     print('assign a new report_file name')
#     breakpoint()
# # Define EVENT directory path
EVENT_DIR = os.path.join(ROOT,'trigger_reports','uw{evid}')
REPORT_FILE = os.path.join(ROOT,'trigger_reports','processing_log_TMP.txt')
if os.path.exists(REPORT_FILE):
    print('assign a new report_file name')
    breakpoint()
# Define EVENT directory path
# Define SAVE directory path
SAVE_FILE = os.path.join(EVENT_DIR,'trigger_report.csv')
SAVE_LABELS = ['P','S','D'] # Leave out 'N' from PhaseNet - not particularly useful
OVERWRITE_PROTECT = True
# Read in full pick catalog
pick_df = pd.read_csv(os.path.join(ROOT,'example','AQMS_event_mag_phase_query_output.csv'))
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x :unix_to_UTCDateTime(x))
print('picks loaded')
evids = list(pick_df.evid.value_counts().index)
# evids

first_write = True

for evid in evids:
    report_file = open(REPORT_FILE, 'a')
    if first_write:
        report_file.write('TIMESTAMP, EVID, KEY, VALUE\n')
        first_write = False
    evid_dir = EVENT_DIR.format(evid=evid)
    print(f'=== STARTING {evid_dir} ===')
    report_file.write(f'{time.time()}, {evid}, START_OF_EVENT_ENTRY, \n')
    report_file.write(f'{time.time()}, {evid}, loading, {evid_dir}\n')
    save_file = SAVE_FILE.format(evid=evid)
    save_path, file = os.path.split(save_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    report_file.write(f'{time.time()}, {evid}, will_write_to, {save_file}\n')
    if OVERWRITE_PROTECT:
        if os.path.isfile(save_file):
            print(f'file: {save_file} already exists - skipping this event and continuing')
            report_file.write(f'{time.time()}, {evid}, SKIP, save_file_exists\n')
            report_file.close()
            continue
    # Subset Picks
    pick_idf = pick_df[pick_df.evid == evid][['evid','orid','arid','net','sta','arrdatetime','iphase','timeres','qual','arrquality']]
    # Convert UNIX times, correcting for leap seconds
    picked_netsta = list(pick_idf[['net','sta']].value_counts().index)
    report_file.write(f'{time.time()}, {evid}, pick_count, {len(pick_idf)}\n')
    tick = time.time()
    ## INIT ##
    # Create copy of the processing line as a full reset for each dataset
    iter_canwyrm = canwyrm.copy()
    
    ## LOAD ##
    # wffile = os.path.join(evid_dir, 'bulk.mseed')
    # st = obspy.read(wffile, fmt='MSEED')
    bulk = []
    breakpoint()
    st = client.get_waveforms_bulk(bulk)
    # for index, row in pick_idf.iterrows()
    ist = obspy.Stream()
    for tr in st:
        # If trace is from a site that has a pick for this event
        if (tr.stats.network, tr.stats.station) in picked_netsta:
            # Place data into float32
            tr.data = tr.data.astype(np.float32)
            # If sampling rate is slightly off, apply resampling correction
            if tr.stats.sampling_rate != round(tr.stats.sampling_rate):
                tr.resample(round(tr.stats.sampling_rate))
            # Append trace to intermediate Stream
            ist += tr
    # Safety catch
    try:
        # Merge data on the ObsPy side
        ist.merge()
    except:
        report_file.write(f'{time.time()}, {evid}, SKIP, bad_stream_merge')
        report_file.close()
        continue
        # Introduce to Wyrm DictStream
    try:
        dst = ds.DictStream(traces=ist)
    except:
        report_file.write(f'{time.time()}, {evid}, SKIP, bad_dictstream_conversion')
        report_file.close()
        continue
    report_file.write(f'{time.time()}, {evid}, trace_count, {len(dst.traces)}\n')
    ## RUN ##
    report_file.write(f'{time.time()}, {evid}, START_WYRM, CanWyrm.pulse\n')
    try:
        can_wyrm_out = iter_canwyrm.pulse(dst)
    except:
        report_file.write(f'{time.time()}, {evid}, SKIP, error_in_pulse')
        report_file.close()
        continue
    report_file.write(f'{time.time()}, {evid}, STOP_WYRM, CanWyrm.pulse()\n')
    report_file.write(f'{time.time()}, {evid}, output_window_count, {sum([len(_v) for _v in can_wyrm_out.values()])}\n')
    tock = time.time()
    print(f'processing for {evid_dir} took {tock - tick:.2f} sec')
    report_file.write(f'{time.time()}, {evid}, starting_output_to, {save_file}\n')

    df_summary = pd.DataFrame()
    try:
        for _k, _v in can_wyrm_out.items():
            print(f'saving {evid} {_k} outputs')
            for _i, idf in enumerate(_v):
                # if _i % 20 == 0:
                #     print(f'{_i+1}/{len(_v)}')
                # idf = _v.popleft()
                # Filter down to desired labels
                idf = idf[idf.label.isin(SAVE_LABELS)]
                # Append EVID to all picks
                idf = idf.assign(evid= [evid for _i in range(len(idf))])
                first_iter = True
                for net, sta in idf[['network','station']].value_counts().index:
                    for label in SAVE_LABELS:
                        ipick_df = pick_idf[(pick_idf.net==net)&(pick_idf.sta==sta)&(pick_idf.iphase==label)]
                        for _, ipick in ipick_df.iterrows():
                            if first_iter:
                                idf = idf.assign(arid=[int(ipick.arid) if irow.label==label else np.nan for idx, irow in idf.iterrows()])
                                first_iter=False
                            else:
                                idf = idf.assign(arid=[int(ipick.arid) if irow.label==label else irow.arid for idx, irow in idf.iterrows()])
                if len(idf) > 0:
                    df_summary = pd.concat([df_summary, idf], axis=0, ignore_index=True)
    except:
        report_file.write(f'{time.time()}, {evid}, SKIP, bad_output_dataframe_compilation')
        report_file.close()
        continue
    try:
        df_summary.to_csv(save_file, header=True, index=False)
        report_file.write(f'{time.time()}, {evid}, saved_to_disk, {save_file}\n')
        report_file.write(f'{time.time()}, {evid}, END_OF_EVENT_ENTRY,\n')
        print(f'saving took {time.time() - tock: .3f}sec')
        report_file.close()
    except:
        report_file.write(f'{time.time()}, {evid}, SKIP, bad_dataframe_write')
        report_file.close()

    # breakpoint()
    ## SAVE ##
    # for _k, _v in can_wyrm_out.items():
    #     # TODO: Finish constructing the termination function
    #     #       GIST: concatenate all trigger lines for a given EVID, split by save_attrs & create dir structure, write chunks into this structure
    #     termination_function(key=_k, queue=_v, base_path=evid_dir, save_attrs = ['network','station','model','weight'])

    # for model, ml_dst_buffer in can_wyrm_out.items():
    #     print(f'saving {model}')
    #     ml_dst_buffer.write(base_path=os.path.join(out_dir, model),
    #                 path_structure='{weight}/{site}')
    #     print(f'saving took {time.time() - tock: .3f}sec')


