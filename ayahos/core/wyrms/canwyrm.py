"""
:module: wyrm.core.wyrm.canwyrm
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:

    CanWyrm - 

"""
from copy import deepcopy
from collections import deque
from ayahos.core.wyrms.wyrm import Wyrm
from ayahos.core.wyrms.tubewyrm import TubeWyrm


class CanWyrm(TubeWyrm):
    """CanWyrm - child class of TubeWyrm - grandchild class of Wyrm

        a submodule class for running a sequence of wyrms' pulse()
        methods in parallel starting with a single input and returning
        a dictionary of each wyrm's output

        id0: {id0: wyrm0.pulse(stdin) = stdout = {id0: wyrm0.output,
              id1: wyrm1.pulse(stdin)             id1: wyrm1.output,
              ...                                    ...
              idN: wyrmN.pulse(stdin)}            idN: wyrmN.output}

    DEV NOTE: 
    
    This class runs items in serial, but with some modification this would
    be a good candidate class for orchestraing multiprocessing.

    Currently the polymorphic pulse() subroutines assume that each pulse
    method does not alter the standard input (stdin). This is friendly with
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
    # _continue_iteration
    # _capture_unit_out
    
    def _get_obj_from_input(self, stdin):
        """_get_obj_from_input for CanWyrm

        return obj = stdin

        :param stdin: standard input object for all wyrms in wyrm_dict
        :type stdin: varies
        :return: deepcopy of stdin
        :rtype: varies
        """        
        obj = stdin
        return obj


    def _unit_process(self, obj):
        """unit_process of CanWyrm

        This unit_process iterates across the wyrms in CanWyrm.wyrm_dict and
        runs their pulse() methods on 

        :param x: _description_
        :type x: _type_
        :param i_: _description_
        :type i_: _type_
        """
        for name, wyrm_ in self.wyrm_dict.items():
            wyrm_.pulse(obj)
            self.logger.debug(f'{name}.output length: {len()}')
        
        unit_out = True
        return unit_out


        


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