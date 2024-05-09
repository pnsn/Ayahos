"""
:module: wyrm.core.wyrm.canwyrm
:auth: Nathan T Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:

    CanWyrm - 

"""
import logging, copy
from wyrm.core.wyrm.wyrm import Wyrm


logger = logging.getLogger(__name__)


class CanWyrm(TubeWyrm):
    """CanWyrm - child class of TubeWyrm - grandchild class of Wyrm

        a submodule class for running a sequence of wyrms' pulse()
        methods in parallel starting with a single input and returning
        a dictionary of each wyrm's output

                y = {id0: wyrm0.pulse(x),
                     id1: wyrm1.pulse(x),
                     ...
                     idN: wyrmN.pulse(x)}

    DEV NOTE: This class runs items in serial, but with some modification
    this would be a good candidate class for orchestraing multiprocessing.
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
        # Handle some basic indexing/formatting
        if not isinstance(wyrm_dict, dict):
            if isinstance(wyrm_dict, (list, tuple)):
                wyrm_dict = dict(zip(range(len(wyrm_dict)), wyrm_dict))
            elif isinstance(wyrm_dict, Wyrm):
                wyrm_dict = {0: wyrm_dict}
            else:
                raise TypeError('wyrm_dict must be type dict, list, tuple, or Wyrm')
        
        # Initialize from Wyrm inheritance
        super().__init__(wyrm_dict=wyrm_dict, wait_sec=wait_sec, max_pulse_size=max_pulse_size)

        self.outputs = {_k: None for _k in self.names}


    def pulse(self, x, **options):
        """
        Triggers the wyrm.pulse(x) method for each wyrm in wyrm_dict, sharing
        the same inputs, and writing outputs to self.dict[wyrmname] via the __iadd__
        method. I.e.,

            self.dict[wyrmname] += wyrm.pulse(x, **options)

        :: INPUTS ::
        :param x: [object] common input for all wyrms in wyrm_dict
        :param options: [kwargs] key word arguments passed to each wyrm's
                    wyrm.pulse(x, **options) method

        :: OUTPUT ::
        :return y: [dict] access to self.dict
        """
        for _i in range(self.max_pulse_size):
            for _k, _v in self.wyrm_dict.items():
                _y = _v.pulse(x, **options)
                # If this wyrm output has not been mapped to self.dict
                if self.dict[_k] is None:
                    self.dict.update({_k: _y})
            if self.debug:
                print(f'CanWyrm pulse {_i + 1}')
            #     for _l, _w in self.dict.items():
            #         print(f'    {_l} - {len(_w)}')
        y = self.dict
        return y

    def __str__(self):
        rstr = f'wyrm.core.coordinate.CanWyrm(wyrm_dict={self.wyrm_dict}, '
        rstr += f'wait_sec={self.wait_sec}, '
        rstr += f'max_pulse_size={self.max_pulse_size}, '
        rstr += f'debug={self.debug})'
        return rstr