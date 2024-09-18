""":module: PULSE.seq.sbm_pipeline
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This provides a wrapper around a sequence of :mod:`~PULSE.module` classes that form a complete
pre-processing, prediction, and post-processing workflow for PULSE using :class:`~SeisBench.models.WaveformModel`-class
machine learning models.

TODO: Code currently commented out. Needs to be completed based on the layout of the "live_example"

"""
# import sys
# from PULSE.module import WindowMod, SequenceMod, InPlaceMod, SeisBenchMod

# class SeisBenchSequenceMod(SequenceMod):
    
#     def __init__(
#             self,
#             model_class_name,
#             weight_names,
#             devicetype='cpu',
#             torch_thread_limit=4,
#             max_batch_size=128,
#             max_prediction_batch_count=5,
#             sampling_rate=100.,
#             sample_overlap=1800,
#             windowing_ref_thresh=0.9,
#             windowing_other_thresh=0.8,
#             channel_fill_rule='cloneZ',
#             max_pulse_size=1024,
#             max_output_size=1e5,
#             report_period=False,
#             meta_memory=60,
#             **kwargs

#     ):
        
#         self.pmethods = ['treat_gaps','sync_to_reference','apply_fill_rule','normalize']

#         if any(_k not in self.pmethods for _k in kwargs.keys()):
#             for _k in kwargs.keys():
#                 if _k not in self.pmethods:            
#                     self.Logger.critical(f'{_k} is not an approved kwarg')
#             self.Logger.critical('exiting')
#             sys.exit(1)
#         sequence = {}
#         # Create SeisBenchMod object
#         seisbenchmod = SeisBenchMod(model_class_name,
#                                     weight_names=weight_names,
#                                     devicetype=devicetype,
#                                     max_batch_size=max_batch_size,
#                                     thread_limit=torch_thread_limit,
#                                     max_pulse_size=max_prediction_batch_count,
#                                     meta_memory=meta_memory,
#                                     max_output_size=max_output_size)
#         # Create Windowing object
#         windowmod = WindowMod(reference_completeness_threshold=windowing_ref_thresh,
#                               other_completeness_threshold=windowing_other_thresh,
#                               model_name=seisbenchmod.model.name,
#                               reference_sampling_rate=sampling_rate,
#                               reference_npts=seisbenchmod.model.in_samples,
#                               max_pulse_size=1,
#                               max_output_size=max_output_size,
#                               meta_memory=60,
#                               report_period=False
#                               )

#         gapsmod = InPlaceMod(
#             pclass='PULSE.data.mlwindow.MLWindow',
#             pmethod='treat_gaps',
#             max_pulse_size=max_pulse_size,
#             max_output_size=max_output_size,
#             meta_memory=meta_memory,
#             report_period=False)


#         # Create Processing Sequences
#         for pmethod in self.pmethods:
#             sequence.update(
#                 {pmethod: InPlaceMod(
#                     pclass='PULSE.data.mlwindow.MLWindow',
#                     pmethod=pmethod,
#                     max_pulse_size=max_pulse_size,
#                     max_output_size=max_output_size,
#                     report_period=False,
#                     meta_memory=meta_memory)
#                 })
        
#         sequence.update({'predict': })

#         for _k, _v in kwargs.items():