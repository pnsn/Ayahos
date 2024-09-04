from collections import deque
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.routing.federator_routing_client import FederatorRoutingClient
from obsplus import WaveBank
from PULSE.module.base import BaseMod
from PULSE.data.dictstream import DictStream
from PULSE.util.pyew import trace2wave
# from PULSE.util.latency import read_latency_file

# TODO: MODULE IN DEVELOPMENT / WORK IN PROGRESS (WIP)


# class ClientMod(BaseMod):
#     def __init__(
#             self,
#             client,
#             max_batch_size=30,
#             max_pulse_size=1,
#             max_output_size=1e5,
#             meta_memory=60,
#             report_period=False
#     ):
#         super().__init__(
#             max_pulse_size=max_pulse_size,
#             max_output_size=max_output_size,
#             meta_memory=meta_memory,
#             report_period=report_period
#         )
#         # Compatability check for client
#         if isinstance(client, (Client, FederatorRoutingClient, WaveBank)):
#             self.client = client        
#         else:
#             self.Logger.critical(f'client is invalid type {type(client)} - supported types: \
#                                   obspy.clients.fdsn.client.Client, \
#                                   obspy.clients.fdsn.routing.federator_routing_client.FederatorRoutingClient, \
#                                   obsplus.bank.wavebank.WaveBank') 
#         if isinstance(max_batch_size, int):
#             if 0 < max_batch_size <= max_pulse_size
            
#     # def _unit_input_from_input(self, input):
#     #     if len(input) < self.max_batch_size:



# class ClientMod(BaseMod):
#     def __init__(self,
#                  client,
#                  latency_file,
#                  starttime=None,
#                  endtime=None,
#                  delayed_start=10,
#                  max_pulse_size=1,
#                  max_output_size=1e10,
#                  report_period=False,
#                  meta_memory=3600):
#         super().__init__(max_pulse_size=max_pulse_size,
#                          max_output_size=max_output_size,
#                          report_period=report_period,
#                          meta_memory=meta_memory)
#         # Compatability check for client
#         if isinstance(client, (Client, FederatorRoutingClient, WaveBank)):
#             self.client = client        
#         else:
#             self.Logger.critical(f'client is invalid type {type(client)} - supported types: \
#                                   obspy.clients.fdsn.client.Client, \
#                                   obspy.clients.fdsn.routing.federator_routing_client.FederatorRoutingClient, \
#                                   obsplus.bank.wavebank.WaveBank')       

# class LatencyPlayerMod(ClientMod):
#     """
#     This module acts in a similar manner to a wave_serverV / tankplayer set of modules for
#     Earthworm in that it provides real-time 
#     """
#     def __init__(self,
#                  client,
#                  latency_file,
#                  starttime=None,
#                  endtime=None,
#                  delayed_start=10,
#                  max_pulse_size=1,
#                  max_output_size=1e10,
#                  report_period=False,
#                  meta_memory=3600):
        
#         super().__init__(
#             client = client,
#             starttime = starttime,
#             endtime=endtime,
#             max_pulse_size=max_pulse_size,
#             max_output_size=max_output_size,
#             report_period=report_period,
#             meta_memory=meta_memory)
        
   

#         # Compatability & formatting check for latency_dataframe
#         latency_dataframe = read_latency_file(latency_file)

#         if isinstance(latency_dataframe, pd.DataFrame):
#             if all(c in self.cols for c in latency_dataframe.columns):
#                 self.index = self._format_check_latency_df(latency_dataframe)


#         if isinstance(starttime, UTCDateTime):
#             self.ref_t0 = starttime
        
#         elif starttime is None:
#             self.ref_t0 = self.index.arrival_time.min() - 5
#         else:
#             self.Logger.critical('provided starttime is not type obspy.core.utcdatetime.UTCDateTime or NoneType')
        
#         if isinstance(endtime, UTCDateTime):
#             self.ref_tf = endtime
        
#         elif endtime is None:
#             self.ref_tf = self.index.arrival_time.max() + 5
#         else:
#             self.Logger.critical('provided endtime is not type obspy.core.utcdatetime.UTCDateTime or NoneType')
    
#         if isinstance(delayed_start, (int, float)):
#             if delayed_start >= 0:
#                 self.delay = float(delayed_start)
#             elif delayed_start is None:
#                 self.delay = 0.
#             else:
#                 self.Logger.warning('Negative delayed_start provided. Setting to 0 sec delay')
#                 self.delay = 0.
#         else:
#             self.Logger.critical('delayed_start is not type int, float or NoneType')
    
#         self._dt = None
#         self._realtime_t0 = None

#     def _measure_input_size(self, input):
#         if self._realtime_t0 is None:
#             self._realtime_t0 = UTCDateTime()
#             self._nowtime = self._realtime_t0
#             self._rectime0 = self.ref_t0 - self.delayed_start
#             self._rectime1 = self.ref_t0
#             self._dt = self._nowtime - self._rectime0
#         else:
#             self._nowtime = UTCDateTime()
#             self._rectime0 = self._rectime1
#             self._rectime1 = self._nowtime - self._dt
        
#         input_size = len(self._current_ldf_view())


#     def _unit_input_from_input(self, input):
#         ldf_view = self._current_ldf_view()
#         bulk = self._ldf_view_to_bulk(ldf_view)
#         unit_input = bulk
#         return unit_input

#     def _should_this_iteration_run(self, input, input_size, iter_number):
#         if input_size > 0:
#             status = True
#         else:
#             status = False
#         return status

#     def _unit_process(self, unit_input):
#         # Get waveform data from client
#         st = self.client.get_waveforms_bulk(unit_input)
#         mlst = DictStream(st)
#         unit_output = deque([])
#         for line in unit_input:
#             mlt = mlst[f'{line[0]}.{line[1]}.{line[2]}.{line[3]}.*'].view_copy(starttime=line[4], endtime=line[5])
#             msg = trace2wave(mlt)
#             unit_output.append(msg)
#         return unit_output
    
#     def _capture_unit_output(self, unit_output):
#         # Append wave messages to output
#         self.output += unit_output
#         # Trim off used values
#         self.index = self.index[(self.index.arrival_time >= self._rectime1)]

#     def _should_next_iteration_run(self, unit_output):
#         """
#         POLYMORPHIC
#         Last updated with :class:`~PULSE.module.client.ClientMod`
        
#         If there are still packet arrival times that post-date the
#         last record time (_rectime1), continue iterations, otherwise
#         trigger early stopping.

#         :param unit_output: _description_
#         :type unit_output: _type_
#         :return: _description_
#         :rtype: _type_
#         """        
#         if self._rectime1 > self.ref_tf:
#             status = False
#         else:
#             status = True
#         return status


#     def _current_ldf_view(self):
#         """Using the time interval [_rectime0, _rectime1), return a view of the
#         latency data frame aliased to the *index* attribute.

#         :return ldf_view: subset view of self.index, or if initial pulse has not been run, a view of the entire index
#         :rtype: pandas.core.dataframe.DataFrame
#         """        
#         if self._realtime_t0 is None:
#             self.Logger.warning('Initial pulse has not been run - showing entire latency dataframe')
#             ldf_view = self.index
#         else:
#             ldf_view = self.index[(self.index.pkt_arrival >= self._rectime0) & (self.index.pkt_arrival < self._rectime1)]
#         return ldf_view