import time, tqdm
quicklog = {'start': time.time()}
import numpy as np
import obspy, os, sys, pandas
sys.path.append(os.path.join('..','..'))
import seisbench.models as sbm
import wyrm.data.dictstream as ds
import wyrm.data.componentstream as cs
import wyrm.core.coordinate as coor
import wyrm.core.process as proc 
from wyrm.util.time import unix_to_UTCDateTime
from collections import deque
import matplotlib.pyplot as plt

quicklog.update({'import': time.time()})
ROOT = os.path.join('..','..','example')
# Load Waveform Data
evid = 61965081
st = obspy.read(os.path.join(ROOT,f'uw{evid}','bulk.mseed'))
# Use Analyst picks to reduce number of assessed
pick_df = pandas.read_csv(os.path.join(ROOT,'AQMS_event_mag_phase_query_output.csv'))
pick_df = pick_df[pick_df.evid == evid][['evid','orid','arid','net','sta','arrdatetime','iphase','timeres','qual','arrquality']]
# Convert UNIX times, correcting for leap seconds
pick_df.arrdatetime = pick_df.arrdatetime.apply(lambda x :unix_to_UTCDateTime(x))
picked_netsta = list(pick_df[['net','sta']].value_counts().index)

ist = obspy.Stream()
for tr in st:
    if (tr.stats.network, tr.stats.station) in picked_netsta:
        ist += tr
# Convert into dst
dst = ds.DictStream(traces=st[:60])#[:60])
# dst.traces = dict(reversed(list(dst.traces.items())))

quicklog.update({'mseed load': time.time()})

# Isolate funny behavior by 'UO.BEER.--.HH?'
# dst = dst.fnselect('UW.AUG.*')
# Set flow volume controls



common_pt = ['stead','instance','iquique','lendb']
EQT = sbm.EQTransformer()
EQT_list = common_pt + ['pnw']
EQT_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN = sbm.PhaseNet()
PN_aliases = {'Z':'Z3','N':'N1','E':'E2'}
PN_list = common_pt + ['diting']
# TODO: Develop extension that mocks up Hydrophone (H)
# PB_aliases = {'Z':'Z3','1':'N1','2':'E2', 'H': 'H4'}
# PBE = sbm.PickBlue(base='eqtransformer')

# PBN = sbm.PickBlue(base='phasenet')

## (ADDITIONAL) DATA SAMPLING HYPERPARAMETERS ##
# reference_sampling_rate 
RSR = 100.
#reference_channel_fill_rule
RCFR= 'zeros'
# Reference component
RCOMP = 'Z'

# Initialize Standard Processing Elements
treat_gap_kwargs = {} # see ComponentStream.treat_gaps() and MLTrace.treat_gaps() for defaults
                      # Essentially, filter 1-45 Hz, linear detrend, resample to 100 sps


# Initialize WindowWyrm elements
windwyrmEQT = proc.WindowWyrm(
    component_aliases=EQT_aliases,
    model_name='EQTransformer',
    reference_sampling_rate=RSR,
    reference_npts=6000,
    reference_overlap=1800,
    pulse_type='site',
    max_pulse_size=64,
    debug=False)

windwyrmPN = proc.WindowWyrm(
    component_aliases=PN_aliases,
    model_name='PhaseNet',
    reference_sampling_rate=RSR,
    reference_npts=3001,
    reference_overlap=900,
    pulse_type='site',
    max_pulse_size=64,
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
    max_pulse_size=10000,
    debug=False)

predwyrmPN = proc.PredictionWyrm(
    model=PN,
    weight_names=PN_list,
    devicetype='mps',
    compiled=False,
    max_pulse_size=10000,
    debug=False)

# Initialize Prediction BufferWyrm elements
pbuffwyrmEQT = coor.BufferWyrm(max_length=200,
                              restrict_past_append=True,
                              blinding=(500,500),
                              method=3,
                              max_pulse_size=10000,
                              debug=False)

pbuffwyrmPN = coor.BufferWyrm(max_length=200,
                             restrict_past_append=True,
                             blinding=(250,250),
                             method=3,
                             max_pulse_size=10000,
                             debug=False)


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
                'predict': predwyrmEQT})#,
                # 'buffer': pbuffwyrmEQT})
    
# Copy/Update to create PhaseNet processing TubeWyrm
tubewyrmPN = tubewyrmEQT.copy().update({'window': windwyrmPN,
                                        'norm': mwyrm_normPN,
                                        'predict': predwyrmPN})#,
                                        # 'buffer': pbuffwyrmPN})

## GROUP TUBES
canwyrm = coor.CanWyrm(wyrm_dict={'EQTransformer': tubewyrmEQT,
                                  'PhaseNet': tubewyrmPN},
                       wait_sec=0,
                       max_pulse_size=30,
                       debug=True)

quicklog.update({'processing initializing': time.time()})
# Execute a single pulse
# tube_wyrm_out = tubewyrmPN.pulse(dst)
can_wyrm_out = canwyrm.pulse(dst)


# # # breakpoint()
# # # Merge pick times with windows of outputs
# # holder = deque()
# # for _dst in tube_wyrm_out:
# #     net = _dst[0].stats.network
# #     sta = _dst[0].stats.station
# #     t0 = _dst.stats.min_starttime
# #     t1 = _dst.stats.max_endtime
# #     dt = _dst[0].stats.delta
# #     _idf = pick_df[(pick_df.net==net) &
# #                    (pick_df.sta==sta) &
# #                    (pick_df.arrdatetime > t0) &
# #                    (pick_df.arrdatetime < t1)]
# #     if len(_idf) > 0:
# #         pick_samples = []
# #         for _i in range(len(_idf)):
# #             pick_time = _idf.arrdatetime.values[_i]
# #             pick_sample = int((pick_time - t0)//dt)
# #             pick_samples.append(pick_sample)
# #         _idf = pandas.concat([_idf, pandas.DataFrame(pick_samples, index=_idf.index, columns=['pick_sample'])], ignore_index=False, axis=1)
# #         holder.append({'dictstream': _dst, 'picks': _idf})

# quicklog.update({'processing complete': time.time()})

# data_frames = {}
# # for _k, tube_wyrm_out in can_wyrm_out.items():
# # Extract Data Window processing information from output MLTrace windows
# holder_incremental = []
# holder_elapsed = []
# # Iterate across windows
# for _y in tube_wyrm_out:
#     # Iterate across traces
#     for _tr in _y:
#         for _i in range(len(_tr.stats.processing) - 1):
#             # Get ID and data starttime
#             line = [_tr.id, _tr.stats.starttime.timestamp]
#             # Get first timestamp and method info
#             line += [_tr.stats.processing[_i][2],
#                     _tr.stats.processing[_i][3],
#                     _tr.stats.processing[_i][0]]
#             # Get second timestamp and method info
#             line += [_tr.stats.processing[_i + 1][2],
#                     _tr.stats.processing[_i + 1][3],
#                     _tr.stats.processing[_i + 1][0]]
#             line += [_tr.stats.processing[_i + 1][0] - _tr.stats.processing[_i][0]]
#             holder_incremental.append(line)
#         line = [_tr.id, _tr.stats.starttime.timestamp]
#         line += [_tr.stats.processing[0][0], _tr.stats.processing[-1][0]]
#         line += [_tr.stats.processing[-1][0] - _tr.stats.processing[0][0]]
#         holder_elapsed.append(line)
# cols = ['ID','t0','module1','method1','stamp1','module2','method2','stamp2','dt21']
# df_inc = pandas.DataFrame(holder_incremental, columns=cols)
# df_tot = pandas.DataFrame(holder_elapsed, columns=['ID','t0','stamp1','stamp2','dt21'])
# # data_frames.update({_k: {'incremental': df_inc, 'total': df_tot}})

# plt.figure()
# plt.subplot(221)
# plt.semilogy(df_inc['stamp1'] - df_inc['stamp1'].min(), df_inc['dt21'],'.')
# # plt.title(f'Model: {_k}')

# # ref_str = '.'.join(df_inc.ID.values[0].split('.')[:-1])
# # IDX = df_inc.ID.str.contains(ref_str)
# # df_ref = df_inc[IDX].sort_values(by='t0')
# # plt.semilogy(df_ref['stamp1'] - df_inc['stamp1'].min(), df_ref['dt21'],'r:')
# plt.xlabel('Runtime from TubeWyrm.pulse(x) execution (sec)')
# plt.ylabel('Incremental Processing Time for Data Windows (sec)')
# plt.subplot(222)
# _i = 0
# x_array = []; y_array = []
# for _y in tube_wyrm_out:
#     for _tr in _y:
#         x_vals = [_p[0] - _tr.stats.processing[0][0] for _p in _tr.stats.processing]
#         y_vals = [_i for _i in range(len(_tr.stats.processing))]
#         x_array.append(x_vals)
#         y_array.append(y_vals)
        
#         plt.step(x_vals,y_vals,'k-', where='post',alpha=0.005)
#         if _i == 0:
#             labels = [_p[3] for _p in _tr.stats.processing]
#             for _i, _l in enumerate(labels):
#                 plt.text(x_vals[_i], y_vals[_i], _l, ha='right', color='red')

# x_array = np.array(x_array)
# y_array = np.array(y_array)

# plt.xlabel('Data Window Processing Time\nRelative to Trim from Buffer (sec)')
# plt.ylabel('Data Window Processing Step Index (#)')

# plt.subplot(223)
# plt.hist(x_array[:, 1:] - x_array[:, :-1], 30, label=labels[1:])
# plt.xlabel('Incremental Processing Time Distribution (sec)')
# plt.ylabel('Data Window Counts')

# plt.subplot(224)
# plt.hist(x_array[:,-1] - x_array[:,0], 100);
# plt.xlabel('Data Window Residence Time (sec)\n[Total time spent in pipeline]')
# plt.ylabel('Data Window Counts')


# plt.show()
# # plt.show()

# # DST_PN = ds.DictStream()
# DST_EQT = ds.DictStream()
# for _k, _v in can_wyrm_out.items():
#     print(_k)
#     for _dst in tqdm.tqdm(_v):
#         for _tr in _dst:
#             if _k == 'PhaseNet':
#                 DST_PN.__add__(_tr.apply_blinding(blinding=(100,100)), key_attr='id', method=3)
#             elif _k == 'EQTransformer':
#                 DST_EQT.__add__(_tr.apply_blinding(blinding=(500,500)), key_attr='id', method=3)

# plt.show()

# for _i in range(len(df_inc)):
#     ser = df_inc.loc[_i, :]
#     plt.text(ser['stamp1'] - df_inc['stamp1'].min(), ser['dt21'],
#              f'{ser["method1"]} - {ser["method2"]}')

# for _m1 in df_inc['method1'].unique():
#     _df = df_inc[df_inc.method1 == _m1]
#     plt.fill_betweenx([1e-4, 1e2],
#                       [_df.stamp1.min() - df_inc.stamp1.min()]*2,
#                       [_df.stamp1.max() - df_inc.stamp1.min()]*2,
#                       alpha=0.1)
#     plt.text(_df.stamp1)