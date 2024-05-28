"""
:module: wyrm.core.wyrm.canwyrm
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:

    CanWyrm - 

"""
import logging
from ayahos.wyrms.tubewyrm import TubeWyrm

Logger = logging.getLogger(__name__)

class CanWyrm(TubeWyrm):
    """CanWyrm - child class of TubeWyrm - grandchild class of Wyrm

        a submodule class for running a sequence of wyrms' pulse()
        methods in parallel starting with a single input and returning
        a dictionary of each wyrm's output

        id0: {id0: wyrm0.pulse(input) = stdout = {id0: wyrm0.output,
              id1: wyrm1.pulse(input)             id1: wyrm1.output,
              ...                                    ...
              idN: wyrmN.pulse(input)}            idN: wyrmN.output}

    DEV NOTE: 
    
    This class runs items in serial, but with some modification this would
    be a good candidate class for orchestraing multiprocessing.

    Currently the polymorphic pulse() subroutines assume that each pulse
    method does not alter the standard input (input). This is friendly with
    wyrm classes like WindowWyrm and CopyWyrm that produce copies (partial 
    or complete) of their inputs without altering the initial input. 
    
    This is antithetic for most other Wyrm classes that use deques and pop/append
    methods as part of the strategy to minimize copying in a given Ayahos module.

    Proceed with this in mind when implementing the CanWyrm class.
    """

    def __init__(self,
                 wyrm_dict,
                 wait_sec=0.,
                 max_pulse_size=1):
        """Initialize a CanWyrm object

        :param wyrm_dict: dictionary or list-like of Wyrm-like objects that accept the same input
            dictionary keys should be succinct names of the different wyrm_dict values 
            - key values are assigned as keys to the `outputs` attribute
            Also accepts lists and tuples of Wyrm-like objects and assigns output keys
            as the position of each Wyrm-like object in wyrm_dict
        :type wyrm_dict: dict, list, or tuple
        :param wait_sec: seconds to wait between execution of elements of wyrm_dict, defaults to 0.
        :type wait_sec: non-negative float-like, optional
        :param max_pulse_size: number of times to send pulse commands to each member of
            wyrm_dict for each pulse this CanWyrm object executes, defaults to 1
        :type max_pulse_size: positive int-like, optional
        """
        # Initialize from Wyrm inheritance
        super().__init__(wyrm_dict=wyrm_dict, wait_sec=wait_sec, max_pulse_size=max_pulse_size)
        # Modify output to be a dictionary with names pulled from the keys of wyrm_dict
        # and alias the outputs form each wyrm to self.output
        self.output = {_k: _v.output for _k, _v in self.wyrm_dict.items()}

    #############################
    # PULSE POLYMORPHIC METHODS #
    #############################   

    # INHERITED FROM TUBEWYRM
    # _should_this_iteration_run
    # _capture_unit_out
    
    def _unit_input_from_input(self, input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.canwyrm.CanWyrm

        return unit_input = input

        :param input: standard input unit_inputect for all wyrms in wyrm_dict
        :type input: varies
        :return: view of input
        :rtype: varies
        """        
        unit_input = input
        return unit_input


    def _unit_process(self, unit_input):
        """
        POLYMORPHIC
        Last updated with :class: `~ayahos.wyrms.canwyrm.CanWyrm`

        Iterate across the wyrms in .wyrm_dict and trigger their .pulse method with unit_input

        :param unit_input: shared unit input object for each wyrm in wyrm_dict.
            NOTE: This input should not be modified by the wyrms in wyrm_dict
                  
        :type unit_input: _type_
        :param i_: _description_
        :type i_: _type_
        """
        nproc = 0
        # Iterate across wyrms in wyrm_dict
        for name, wyrm_ in self.wyrm_dict.items():
            y, inproc = wyrm_.pulse(
                unit_input,
                mute_logging=self.mute_interal_logging)
            if self.mute_internal_logging:
                Logger.debug(f'{name}: {inproc}')
            nproc += inproc
        unit_output = nproc
        return unit_output

    # def pulse(self, x, **options):
    #     """
    #     Triggers the wyrm.pulse(x) method for each wyrm in wyrm_dict, sharing
    #     the same inputs, and writing outputs to self.dict[wyrmname] via the __iadd__
    #     method. I.e.,

    #         self.dict[wyrmname] += wyrm.pulse(x, **options)

    #     :: INPUTS ::
    #     :param x: [object] common input for all wyrms in wyrm_dict
    #     :param options: [kwargs] key word arguments passed to each wyrm's
    #                 wyrm.pulse(x, **options) method

    #     :: OUTPUT ::
    #     :return y: [dict] access to self.dict
    #     """
    #     for _i in range(self.max_pulse_size):
    #         for _k, _v in self.wyrm_dict.items():
    #             _y = _v.pulse(x, **options)
    #             # If this wyrm output has not been mapped to self.dict
    #             if self.dict[_k] is None:
    #                 self.dict.update({_k: _y})
    #         if self.debug:
    #             print(f'CanWyrm pulse {_i + 1}')
    #         #     for _l, _w in self.dict.items():
    #         #         print(f'    {_l} - {len(_w)}')
    #     y = self.dict
    #     return y

    # def __str__(self):
    #     rstr = f'wyrm.core.coordinate.CanWyrm(wyrm_dict={self.wyrm_dict}, '
    #     rstr += f'wait_sec={self.wait_sec}, '
    #     rstr += f'max_pulse_size={self.max_pulse_size}, '
    #     rstr += f'debug={self.debug})'
    #     return rstr