"""
:module: wyrm.coordinating.sequence
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains class defining submodules for coordinating sets
    of wyrms to form parts of fully operational Wyrm modules, including:

    TubeWyrm - a submodule class for running a sequence of wyrms' pulse()
              methods in series starting with a single input to the first
              wyrm in the series and providing a single output from the
              last wyrm in the series

              y = wyrmN.pulse(...pulse(wyrm1.pulse(wyrm0.pulse(x))))

TODO: Turn status from _capture_unit_out into a representation of nproc
    idea is to say "keep going" if the tubewyrm is conducting any processes

"""
import logging
import numpy as np
import pandas as pd
from collections import deque
from ayahos.wyrms.wyrm import Wyrm


Logger = logging.getLogger(__name__)

class TubeWyrm(Wyrm):
    """
    Wyrm child-class facilitating chained execution of pulse(x) class methods
    for a sequence wyrm unit_inputects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input.

    The pulse method operates as follows
    for wyrm_dict = {key0: <wyrm0>, key2: <wyrm2> , key1: <wyrm1>}
        tubewyrm.pulse(x) = wyrm1.pulse(wyrm2.pulse(wyrm0.pulse(x)))

        Note that the dictionary ordering dictates the execution order!
    """

    def __init__(
            self,
            wyrm_dict={},
            log_pulse_summary=True,
            wait_sec=0.0,
            max_pulse_size=1,
            meta_memory=3600,
            report_period=None):
        """
        Create a TubeWyrm unit_inputect
        :param wyrm_dict: collection of [Wyrm-type] unit_inputects that are executed in their provided order. 
                    dict-type entries allow for naming of component wyrms for user-friendly assessment. list, deque, and Wyrm-type
                    all other input types use a 0-indexed key.
        :type wyrm_dict: dict, list, or tuple
        :param wait_sec: seconds to wait between execution of each Wyrm in wyrm_dict, default is 0.0
        :type wait_sec: float
        :param max_pulse_size: number of times to run the sequence of pulses, default is 1
        :type max_pulse_size: int
        """
        # Inherit from Wyrm
        super().__init__(max_pulse_size=max_pulse_size,
                         meta_memory=meta_memory,
                         report_period=report_period)
        if isinstance(log_pulse_summary, bool):
            self.log_pulse_summary = log_pulse_summary
        # wyrm_dict compat. checks
        if isinstance(wyrm_dict, Wyrm):
            self.wyrm_dict = {0: wyrm_dict}
        elif isinstance(wyrm_dict, dict):
            if all(isinstance(_w, Wyrm) for _w in wyrm_dict.values()):
                self.wyrm_dict = wyrm_dict
            else:
                raise TypeError('All elements in wyrm_dict must be type Wyrm')
            
        # Handle case where a list or deque are provided
        elif isinstance(wyrm_dict, (list, deque)):
            if all(isinstance(_w, Wyrm) for _w in wyrm_dict):
                # Default keys are the sequence index of a given wyrm
                self.wyrm_dict = {_k: _v for _k, _v in enumerate(wyrm_dict)}
            else:
                raise TypeError('All elements in wyrm_dict must be type Wyrm')
        else:
            raise TypeError('wyrm_dict must be a single Wyrm-type unit_input or a list or dictionary thereof')
        
        # wait_sec compat. checks
        if not isinstance(wait_sec, (int, float)):
            raise TypeError('wait_sec must be float-like')
        elif wait_sec < 0:
            raise ValueError('wait_sec must be non-negative')
        else:
            self.wait_sec = wait_sec
        # Create a list representation of keys
        self.names = list(wyrm_dict.keys())
        # Alias the output of the last wyrm in wyrm_dict to self.output (inherited from Wyrm)
        if len(self.wyrm_dict) > 0:
            self._alias_wyrm_dict_output()



    def update(self, new_dict):
        """
        Apply update to wyrm_dict using the 
        dict.update(new_dict) builtin_function_or_method
        and then update relevant attributes of this TubeWyrm

        This will update existing keyed entries and append new
        at the end of self.wyrm_dict (same behavior as dict.update)

        :param new_dict: dictionary of Wyrm-like unit_input
        :type new_dict: dict
        :updates: 
            - **self.wyrm_dict** - updates wyrm_dict as specified
            - **self.names** - updates the list of keys in wyrm_dict
            - **self.output** - re-aliases to the output attribute of the last wyrm in wyrm_dict

        """
        # Safety catches identical to those in __init__
        if not isinstance(new_dict, dict):
            raise TypeError('new_dict must be type dict')
        elif not all(isinstance(_w, Wyrm) for _w in new_dict.values()):
            raise TypeError('new_dict can only have values of type Wyrm')
        else:
            pass
        # Run updates on wyrm_dict
        self.wyrm_dict.update(new_dict)
        # Update names attribute
        self.names = list(self.wyrm_dict.keys())
        # Set outputs as alias to last wyrm
        self._alias_wyrm_dict_output()


    def _alias_wyrm_dict_output(self):
        """
        Alias the self.output attribute of the last wyrm-type unit_inputect
        in this TubeWyrm's wyrm_dict to this TubeWyrm's output attribute

        i.e., 
        TubeWyrm.wyrm_dict = {0: wyrm0, 1: wyrm1, 2: wyrm2}
        TubeWyrm.output = wyrm2.output
        """        
        self.output = list(self.wyrm_dict.values())[-1].output
    
    def remove(self, key):
        """
        Convenience wrapper of the dict.pop() method
        to remove an element from self.wyrm_dict and
        associated attributes

        :: INPUT ::
        :param key: valid key in self.wyrm_dict.keys()
        :type key: unit_inputect

        :: RETURN ::
        :return popped_item: popped (key, value) pair
        :rtype popped_item: tuple
        """
        if key not in self.wyrm_dict.keys():
            raise KeyError(f'key {key} is not in self.wyrm_dict.keys()')
        # Remove key/val combination from dict
        val = self.wyrm_dict.pop(key)
        # Update names attribute
        self.names = list(self.wyrm_dict.keys())
        # Return key and value
        self._alias_wyrm_dict_output()
        return (key, val)

    def reorder(self, reorder_list):
        """
        Reorder the current contents of wyrm_dict using either
        an ordered list of wyrm_dict

        :: INPUT ::
        :param reorder_list: unique list of keys from self.wyrm
        :type reorder_list: list of Wyrm-likes
        """
        # Ensure reorder_list is a list
        if not isinstance(reorder_list, list):
            raise TypeError('reorder_list must be type list')

        # Ensure reorder_list is a unique set
        tmp_in = []
        for _e in reorder_list:
            if _e not in tmp_in:
                tmp_in.append(_e)
        if tmp_in != reorder_list:
            raise ValueError('reorder_list has repeat entries - all entries must be unique')

        # Conduct reordering if checks are passed
        # Handle input (re)ordered wyrm_dict key list
        if all(_e in self.wyrm_dict.keys() for _e in reorder_list):
            tmp = {_e: self.wyrm_dict[_e] for _e in reorder_list}
        # Handle input (re)ordered index list
        elif all(_e in np.arange(0, len(reorder_list)) for _e in reorder_list):
            tmp_keys = list(self.wyrm_dict.keys())
            tmp = {_k: self.wyrm_dict[_k] for _k in tmp_keys}

        # Run updates
        self.wyrm_dict = tmp
        self.names = list(tmp.keys())
        self._alias_wyrm_dict_output()


    def __repr__(self, extended=False):
        """Provide a user-friendly summary of the contents of this TubeWyrm
        :: INPUT ::
        :param extended: show full __repr__ output of component Wyrms? , defaults to False
        :type extended: bool, optional

        :: OUTPUT ::
        :return rstr: string representation of this Wyrm's contents
        :rtype rstr: str
        """
        rstr = f'{super().__repr__()}\n'
        rstr = f"(wait: {self.wait_sec} sec)\n"
        for _i, (_k, _v) in enumerate(self.wyrm_dict.items()):
            # Provide index number
            rstr += f'({_i:<2}) '
            # Provide labeling of order
            if _i == 0:
                rstr += "(head) "
            elif _i == len(self.wyrm_dict) - 1:
                rstr += "(tail) "
            else:
                rstr += "  ||   "
            rstr += f"{_k} | "
            if extended:
                rstr += f'{_v.__str__()}\n'
            else:
                rstr += f'{type(_v)}\n'
        return rstr
    
    #############################
    # PULSE POLYMORPHIC METHODS #
    #############################

    def _should_this_iteration_run(self, input, input_measure, iterno):
        """
        POLYMORPHIC

        Last updated with :class: `~ayahos.wyrms.tubewyrm.TubeWyrm`

        always return status = True
        Execute max_pulse_size iterations regardless of internal processes

        :param input: Unused
        :type input: any
        :param iterno: Unused
        :type iterno: any
        :return status: continue iteration - always True
        :rtype: bool
        """        
        status = True
        return status
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.tubewyrm.TubeWyrm` 

        Pass the standard input directly to the first wyrm in wyrm_dict

        :param input: standard input unit_inputect
        :type input: varies, depending on input expected by first wyrm in wyrm_dict
        :return unit_input: view of standard input unit_inputect
        :rtype unit_input: varies
        """        
        unit_input = input
        return unit_input

    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.tubewyrm.TubeWyrm`

        Chain pulse() methods of wyrms in wyrm_dict
        passing unit_input as the input to the first 

            wyrm2.pulse(wyrm1.pulse(wyrm0.pulse(unit_input)))

        In this example output is captured by wyrm2's pulse and
        TubeWyrm.output is a view of wyrm2.output

        :param unit_input: standard input for the pulse method of the first wyrm in wyrm_dict
        :type unit_input: varies
        :return unit_output: sum of nproc output by each wyrm in wyrmdict for this iteration
        :rtype: int
        """ 
        for j_, wyrm_ in enumerate(self.wyrm_dict.values()):
            if j_ == 0:
                y = wyrm_.pulse(unit_input)
            else:
                y = wyrm_.pulse(y)
        # TODO: Placeholder unconditional True - eventually want to pass a True/False
        # up the line to say if there should be early stopping
        unit_output = None
        return unit_output

    def _capture_unit_output(self, unit_output):
        """
        POLYMORPHIC
        Last updated by :class: `~ayahos.wyrms.tubewyrm.TubeWyrm`

        Termination point - output capture is handled by the last
        Wyrm-Type object in wyrm_dict and aliased to this TubeWyrm's
        output attribute.

        :param unit_output: sum of processes executed by the _unit_process call
        :type unit_output: int
        :return: None
        :rtype: None
        """
        return None   

    def _should_next_iteration_run(self, unit_output):
        """
        POLYMORPHIC
        Last updated by :class: `~ayahos.wyrms.tubewyrm.TubeWyrm`

        Signal early stopping (status = False) if unit_output == 0

        :param unit_output: number of processes executed by _unit_process
            Sum of pulse 
        :type unit_output: int
        :return status: should the next iteration be run?
        :rtype status: bool
        """
        if unit_output == 0:
            status = False
        else:
            status = True
        return status

    def _update_report(self):
        """
        POLYMORPHIC
        Last updated with :class:`~ayahos.wyrms.tubewyrm.TubeWyrm`

        Get the mean value line for each wyrm and add information
        on the pulserate, number of logged pulses, and memory period
        for each wyrm.
        """
        report_dict = {}
        for _n, _w in self.wyrm_dict.items():
            _r = _w.report
            _p = _w._pulse_rate
            for _m in ['last','mean']:
                line = list(_r.loc[_m].values)
                line.append(_p)
                line.append(len(_w._metadata))
                line.append(_w.meta_memory)
                report_dict.update({f'{_n}({_m})': line})
        keys = self._keys_meta + ['p_rate','n_pulse','memory_sec']
        self.report = pd.DataFrame(report_dict, index=keys).T
        self.report.index.name = 'submod (stat)'



    # def unit_process(self, x):
    #     """unit_process for ayahos.core.wyrms.tubewyrm.TubeWyrm

    #     Execute a chained pulse of wyrm-type unit_inputects in the self.wyrm_dict

    #     i.e., wyrm_dict = {0: wyrm1, 1: wyrm2}
    #     y = wyrm2.pulse(wyrm1.pulse(x))

    #     Iterate for max_pulse_size
    #         Iterate across wyrms in self.wyrm_dict
    #             If first wyrm_, y = wyrm_.pulse(x)
    #             else: y = wyrm_pulse(y)
        
    #             NOTE: No _continue_iteration check used ehre
        
    #     :param x: input expected by first wyrm-type unit_inputects' wyrm_.pulse() method in the self.wyrm_dict
    #     :type x: varies, typically collections.deque or ayahos.core.streamdictstream.DictStream
    #     :return y: output from wyrm_.pulse() for the last wyrm-type unit_inputect in the self.wyrm_dict
    #     :type y: varies, typically collections.deque or ayahos.core.stream.dictstream.DictStream
        
    #     """        
    #     for j_, (name, wyrm_) in enumerate(self.wyrm_dict.items()):
    #         if j_ == 0:
    #             y = wyrm_.pulse(x)
    #             Logger.debug(f'{name}.output length {len}')
    #         else:
    #             y = wyrm_.pulse(y)
    #     return True

    # def pulse(self, x):
    #     """pulse method for ayahos.core.wyrms.tubewyrm.TubeWyrm

    #     Pulse method is a polymorphic execution of Wyrm.pulse()
    #     see TubeWyrm.core_process() for details

    #     :param x: collection of input unit_inputects for the first wyrm in TubeWyrm.wyrm_dict
    #     :type x: varies
    #     :return y: alias to the output attribute of the last wyrm in TubeWyrm.wyrm_dict
    #     :rtype y: varies
    #     """
    #     y = super().pulse(x)
    #     return y

    # def pulse(self, x):
    #     """
    #     Initiate a chained pulse for elements of wyrm_queue.

    #     E.g.,
    #     tubewyrm.wyrm_dict = {name0:<wyrm0>,
    #                           name1:<wyrm1>,
    #                           name2:<wyrm3>}
    #     y = tubewyrm.pulse(x)
    #         is equivalent to
    #     y = wyrmw.pulse(wyrmq.pulse(wyrm0.pulse(x)))

    #     Between each successive wyrm in the wyrm_dict there
    #     is a pause of self.wait_sec seconds.

    #     :: INPUT ::
    #     :param x: Input `x` for the first Wyrm unit_inputect in wyrm_dict

    #     :: OUTPUT ::
    #     :param y: Output `y` from the last Wyrm unit_inputect in wyrm_dict
    #     """
    #     if self.debug:
    #         start = time.time()
    #     out_lens = []
    #     for _i in range(self.max_pulse_size):
    #         if self.debug:
    #             print(f'TubeWyrm pulse {_i} - {time.time() - start:.3e}sec')
    #         for _j, _wyrm in enumerate(self.wyrm_dict.values()):
    #             if self.debug:
    #                 print('Tube∂ Pulse Element Firing')
    #                 print(f'   {_wyrm}')
    #                 tick = time.time()
                    
    #             # For first stage of pulse, pass output to `y`
    #             if _j == 0:
    #                 y = _wyrm.pulse(x)
    #                     # print(f' ----- {len(y)} elements coming out')
    #                 out_lens.append(len(y))

    #             # For all subsequent pulses, update `y`
    #             else:
    #                 # if self.debug:
    #                     # print(f' ----- {len(y)} elements going in')
    #                 y = _wyrm.pulse(y)
    #                 # if self.debug:
    #                     # print(f' ----- {len(y)} elements coming out')
    #                 out_lens.append(len(y))
    #             if self.debug:
    #                 print(f'    Pulse Element Runtime {time.time() - tick:.3f}sec')
    #                 print(f'    Elapsed Pulse Time {tick - start:.3f}sec\n')

    #             # if not last step, wait specified wait_sec
    #             if _j + 1 < len(self.wyrm_dict):
    #                 time.sleep(self.wait_sec)
    #         # if self.debug:
    #         for _k, _v in zip(self.wyrm_dict.keys(), out_lens):
    #             print(f'{_k} output length: {_v}')
    #     return y

