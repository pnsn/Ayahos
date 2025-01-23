"""
:module: PULSE.mod.unit.process
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This contains class definitions for data processing unit modules that either
    conduct in-place alterations to data objects or capture outputs of data objects'
    class-method outputs

# TODO: Need to write pulse subroutines with simplified structure

"""

import sys, os
from PULSE.mod.base import BaseMod

# Logger = logging.getLogger(__name__)

class ProcMod(BaseMod):
    """
    A class for executing a specified class method on objects that enacts in-place changes on input objects and
    passes the altered objects to the **ProcMod.output** attribute. At it's core this is conducted using the
    :meth:`~getattr` method composed as:

        getattr(obj, pmethod)(**pkwargs)

    :param pclass: name of the class, with full import extension, on which this module will operate.
    :type pclass: str
    :param pmethod: name of the class method that this module will execute.
    :type pmethod: str
    :param pkwargs: arguments (positional and kwarg), formatted as key-word arguments, to pass to **pmethod**, defaults to {}.
    :type pkwargs: dict, optional.
    :param max_pulse_size: maximum number of iterations for this object's :meth:`~PULSE.mod.process.ProcMod.pulse` method, defaults to 10000.
    :type max_pulse_size: int, optional.
    :param maxlen: maximum length of the **output** :class:`~collections.deque` attribute for this object, defaults to None. 
    :type maxlen: int or NoneType, optional.
    :param name: string to append to the end of this object's __name__, defaults to None.
    :type name: NoneType or str, optional.


    .. rubric:: Filtering an obspy trace
    >>> from PULSE.mod.process import ProcMod
    >>> from collections import deque
    >>> from obspy import read
    >>> st = read()
    >>> inpt = deque([tr for tr in st])
    >>> inpt[0].stats.processing
    []
    >>> ipmod = ProcMod('obspy.core.trace.Trace','filter',{'type':'bandpass','freqmin':1,'freqmax':20})
    >>> output = ipmod.pulse(inpt)
    >>> output[0].stats.processing
    ["ObsPy 1.4.1: filter(options={'freqmin': 1, 'freqmax': 20}::type='bandpass')"]
    

    This class introduces the :meth:`~PULSE.mod.process.
    
    Initialization Notes
    - sanity checks are applied to ensure that pmethod is in the attributes and methods associated with pclass
    - the only sanity check applied is that pkwargs is type dict. 
    Users should refer to the documentation of their intended pclass.pmethod() to ensure keys and values are compatable.
    
    
    """
    def __init__(
        self,
        pclass,
        pmethod,
        pkwargs={},
        mode='inplace',
        max_pulse_size=10000,
        maxlen=None,
        name=None
        ):
        """
        Initialize a :class:`~PULSE.mod.process.ProcMod` object

        :param pclass: full import path and name of class this ProcMod will operate on
        :type pclass: str, e.g., "PULSE.data.window.Window"
        :param pmethod: name of class method to apply to unit_input objects
        :type pmethod: str, e.g., "filter"
        :param pkwargs: key-word arguments (and positional arguments stated as key-word arguments) for pclass.pmethod(**pkwargs)
        :type pkwargs: dict, e.g., {"type": "bandpass", "freqmin": 1, "freqmax": 45}
        :param max_pulse_size: maximum number of iterations to conduct in a pulse, defaults to 10000.
        :type max_pulse_size: int, optional
        """

        # Initialize/inherit from BaseMod
        super().__init__(max_pulse_size=max_pulse_size, maxlen=maxlen, name=None)

        # pclass compatability checks
        self.pclass = self.import_class(pclass)
        # pmethod compatability checks
        if pmethod not in [func for func in dir(self.pclass) if callable(getattr(self.pclass, func))]:
            msg = f'pmethod "{pmethod}" is not defined in {self.pclass} properties or methods. '
            raise AttributeError(msg)
            # self.Logger.critical(msg)
        else:
            self.pmethod = pmethod
        # pkwargs compatability checks
        if isinstance(pkwargs, dict):
            self.pkwargs = pkwargs
        else:
            raise TypeError(f'input "pkwargs" must be type dict. Exiting')

        # Compatability check for mode
        if mode not in ['inplace','output']:
            raise ValueError(f'input "mode = {mode}" not supported. Exiting.')
        else:
            self.mode = mode
        if name is None:
            snstr = f'{self.mode}_{self.pmethod}'
        else:
            snstr = f'{name}_{self.mode}_{self.pmethod}'
        self.setname(snstr)

    def get_unit_input(self, input):
        """POLYMORPHIC METHOD

        :class:`~PULSE.mod.process.ProcMod` extends the inherited method :meth:`~PULSE.mod.base.BaseMod.get_unit_input`
        adding a safety check that the object popped off of **input** is type **ProcMod.pclass**.

        :param input: collection of **pclass**-type objects
        :type input: collections.deque
        :return:
         - **unit_input** (*pclass*) -- object to be modified in :meth:`~PULSE.mod.process.ProcMod.run_unit_process`
        """        
        # Use checks from BaseMod on input
        unit_input = super().get_unit_input(input)
        if unit_input is None:
            return None
        # Then apply checks from pclass
        if isinstance(unit_input, self.pclass):
            return unit_input
        else:
            self.Logger.critical(f'object popped from input mismatch {self.pclass} != {type(unit_input)}. Exiting on DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)

        
    def run_unit_process(self, unit_input):
        """POLYMORPHIC METHOD

        Last updated with :class:`~PULSE.mod.process.ProcMod`

        Execute the specified class-method (pmethod) with specified arguments (pkwargs)
        on an unit_input object of type pclass.


        :param unit_input: object to be modified
        :type unit_input: pclass
        :returns:
         - **unit_output** (*ProcMod.pclass*) -- modified object
        """ 
        try:
            if self.mode == 'inplace':
                getattr(unit_input, self.pmethod)(**self.pkwargs)
                unit_output = unit_input
            elif self.mode == 'output':
                unit_output = getattr(unit_input, self.pmethod)(**self.pkwargs)
        except:
            breakpoint()
            self.Logger.critical(f'{self.pmethod} did not work on unit_input of type {type(unit_input)}. Exiting on DATAERR ({os.EX_DATAERR})')
            sys.exit(os.EX_DATAERR)
        return unit_output
    


    #     # TODO: Processing steps should be handled by the data class object, not the module class object
    #     # except:
    #     #      self.Logger.warning(f'{self.pmethod} did not work on unit input - skipping')
    #     #      unit_output = None
    #     #     return unit_output
    #     # if self.pclass in [MLTrace, DictStream, Window]:
    #     #     unit_input.stats.processing.append([self.__name__(full=False), self.pmethod, UTCDateTime()])
    #     # return unit_output
    
    # def capture_unit_output(self, unit_output):
    #     super().capture_unit_output(unit_output)


    

# class OutputMod(ProcMod):
#     """A child class of ProcMod that orchestrates execution of a class method for
#     input data objects and captures their outputs in the OutputMod.output attribute.

#     .. rubric:: Creating copies of DictStreams at a rate of <= 20 per pulse
#     >>> from PULSE.mod.process import OutputMod
#     >>> from PULSE.data.dictstream import DictStream
#     >>> from collections import deque
#     >>> inpt = deque([DictStream() for x in range(40)])
#     >>> len(inpt)
#     40
#     >>> outmod = OutputMod(
#             pclass='PULSE.data.dictstream.DictStream',
#             oclass='PULSE.data.dictstream.DictStream',
#             pmethod='copy',
#             pkwargs={},
#             max_pulse_size=20)
#     >>> output = outmod.pulse(inpt)
#     >>> output
#     >>> outmod
    
#     :param pclass: string-formatted import path of the class expected for unit_input objects
#     :type pclass: str
#     :param oclass: string-formatted import path of the class expected for unit_output objects
#     :type oclass: str
#     :param pmethod: name of class method for **pclass** to execute
#     :type pmethod: str
#     :param pkwargs: key word arguments to pass as **pclass.pmethod(\*\*pkwargs)**, defaults to {}.
#     :type pkwargs: dict, optional
#     :param delete_processed_inputs: should unit_input objects be deleted after rendering a unit_output in :meth:`~PULSE.mod.process.OutputMod.run_unit_process`? Defaults to True
#     :type delete_processed_inputs: bool, optional
#     :param max_pulse_size: maximum number of input objects to process per pulse, defaults to 10000
#     :type max_pulse_size: int, optional
#     :param maxlen: maximum length of the :class:`~collections.deque` **OutputMod.output** attribute, defaults to None.
#     :type maxlen: int or NoneType, optional
#     :param name: string to append to the end of this object's __name__ attribute, defaults to None.
#     :type name: NoneType or str, optional
#     """    
#     def __init__(
#             self,
#             pclass,
#             oclass,
#             pmethod,
#             pkwargs={},
#             delete_processed_inputs=True,
#             max_pulse_size=10000,
#             maxlen=None,
#             name=None):
#         """Initialize an :class:`~PULSE.mod.process.OutputMod` object
        
#         :param pclass: string-formatted import path of the class expected for unit_input objects
#         :type pclass: str
#         :param oclass: string-formatted import path of the class expected for unit_output objects
#         :type oclass: str
#         :param pmethod: name of class method for **pclass** to execute
#         :type pmethod: str
#         :param pkwargs: key word arguments to pass as **pclass.pmethod(\*\*pkwargs)**, defaults to {}.
#         :type pkwargs: dict, optional
#         :param delete_processed_inputs: should unit_input objects be deleted after rendering a unit_output in :meth:`~PULSE.mod.process.OutputMod.run_unit_process`? Defaults to True
#         :type delete_processed_inputs: bool, optional
#         :param max_pulse_size: maximum number of input objects to process per pulse, defaults to 10000
#         :type max_pulse_size: int, optional
#         :param maxlen: maximum length of the :class:`~collections.deque` **OutputMod.output** attribute, defaults to None.
#         :type maxlen: int or NoneType, optional
#         :param name: string to append to the end of this object's __name__ attribute, defaults to None.
#         :type name: NoneType or str, optional
#         """        
#         super().__init__(
#             pclass=pclass,
#             pmethod=pmethod,
#             pkwargs=pkwargs,
#             max_pulse_size=max_pulse_size,
#             maxlen=maxlen,
#             name=name)
        
#         if not isinstance(oclass, str):
#             self.Logger.critical(f'oclass must be type str. Not {type(oclass)}. Exiting on DATAERR ({os.EX_DATAERR})')
#             sys.exit(os.EX_DATAERR)
#         else:
#             # In the event the output class is a numeric or string class
#             if oclass in ['str','int','float']:
#                 self.oclass = eval(oclass)
#             # In all other cases
#             else:
#                 self.oclass = self.import_class(oclass)
        
#         if not isinstance(delete_processed_inputs, bool):
#             self.Logger.critical('')
    
#     def measure_input(self, input):
#         return super().measure_input(input)
    
#     def measure_output(self):
#         return super().measure_output()
    
#     def get_unit_input(self, input):
#         return super().get_unit_input(input)
    
#     def run_unit_process(self, unit_input):
#         """POLYMORPHIC METHOD

#         Last updated with :class:`~PULSE.mod.process.OutputMod`

#         Executes the specified class method of unit_input with input arguments provided
#         when this module was initialized:

#         unit_output = getattr(unit_input, pmethod)(\*\*pkwargs)

#         If delete_processed_inputs is True, then **unit_input** is deleted from
#         memory after successful execution of the class method but before unit_output
#         is returned. This is useful in preventing build-up of abandoned in-memory objects.
        
#         :param unit_input: object 
#         :type unit_input: _type_
#         :return: _description_
#         :rtype: _type_
#         """        
#         try:
#             unit_output = getattr(unit_input, self.pmethod)(**self.pkwargs)
#         except:
#             self.Logger.critical(f'{self.pmethod} did not work on unit_input of type {type(unit_input)}. Exiting on DATAERR ({os.EX_DATAERR})')
#             sys.exit(os.EX_DATAERR)
#         if self.delete_processed_inputs:
#             del unit_input
#         return unit_output
    
#     def capture_unit_output(self, unit_output):
#         """POLYMORPHIC METHOD

#         :class:`~PULSE.mod.process.OutputMod` uses the inherited :meth:`~PULSE.mod.process.ProcMod.capture_unit_output` method

#         :param unit_output: output of **pclass.pmethod(\*\*pkwargs)**
#         :type unit_output: _type_
#         """        
#         super().capture_unit_output(unit_output)


    # def __str__(self):
    #     rstr = f'{self.__class__.__name__}\n'
    #     rstr += f'{self.pclass.__name__}.{self.pmethod} --> {self.oclass.__name__}'
    #     return rstr