import os, sys, logging
import seisbench.models as sbm
sys.path.append(os.path.join('..', '..'))
from wyrm.core.coordinate import TubeWyrm
from wyrm.core.process import WindowWyrm, ProcWyrm, WaveformModelWyrm
from wyrm.core.data import InstrumentWindow, BufferTree, TraceBuffer

# Initialize Module Logger
module_logger = logging.getLogger(__name__)


# ML_TUBEWYRM CLASS DEFINITION #

class ML_TubeWyrm(TubeWyrm):
    """
    The ML_TubeWyrm class provides:
     
    1)  a composed version of the wyrm.core.coordinate.TubeWyrm class that
        contains the following processing sequence (component wyrms, hereafter)

            WindowWyrm -> ProcWyrm -> WaveformModelWyrm
    
    2)  a concise set of input key-word arguments that allow users to largely leverage pre-
        defined kwargs for each component, while still providing access to detailed
        parameter tuning via the ??_kwargs key-word arguments

    3)  logging functionality - NOTE: this will be extended to the TubeWyrm
        and other component wyrm classes in future revisions.
    
    This allows syntactically simple composition of a complete Wyrm module
    by using (minimally) input and output RingWyrm, and a HeartWyrm, objects
    to coordinate waveform reading, output writing, and module operation.

    """
    def __init__(
        self,
        model=sbm.EQTransformer(),
        weight_names=['pnw','instance','stead'],
        ww_kwargs={'completeness':{'Z':0.95, 'N':0.95, 'E':0.95}},
        pw_kwargs={'class_method_list': ['.default_processing(max_length=0.06)']},
        ml_kwargs={'devicetype': 'mps'},
        recursive_debug=False,
        wait_sec=0.,
        max_pulse_size=1,
        debug=False
        ):

        """
        Initialize a ML_TubeWyrm submodule object

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel] WaveformModel child-class object
                        (e.g. seisbench.models.EQTransformer()) that documents a Neural
                        Network architecture
        :param weight_names: [list-like] of [str] names of pretrained model weights
                        compatable with 'model'
        :param ww_kwargs: [dict] dictionary of key-word arguments to overwrite standard
                        kwargs for wyrm.core.process.WindowWyrm.__init__. The following
                        parameters are overwritten by the contents of `model`
                            target_samprate = model.sampling_rate
                            target_overlap  = model._annotate_args['overlap'][1]
                            target_blinding = model._annotate_args['blinding'][1][0]
                            target_order    = model.component_order
        :param pw_kwargs: [dict] dictionary of key-word arguments to overwrite standard
                        kwargs for wyrm.core.process.ProcWyrm.__init__.
        :param ml_kwargs: [dict] dictionary of key-word arguments to overwrite standard
                        kwargs for wyrm.core.process.WaveformModelWyrm.__init__.
                        The following parameters are inherited/overwritten from information
                        in `model`:
                            model = model
                            weight_names = weight_names (with checks against model)
                            max_samples = 2*model.in_samples (if LHS < RHS)
        :param recursive_debug: [bool] force all component wyrms to have the same 
                            debug status as this ML_TubeWyrm?
                                True - Yes
                                False - No
        :param wait_sec: [float] pause time between execution of component wyrms
                        also see wyrm.core.coordinate.TubeWyrm
        :param max_pulse_size: [int] number of times to run the core process in pulse()
                        per time pulse() is called
        :param debug: [bool] run this ML_TubeWyrm in debug mode?
                        True - more detailed logging
                        False - raise Errors rather than logging 
        """
        # initialize submodule logger
        self.logger = logging.getLogger('wyrm.module.subtubes.ML_TubeWyrm')
        self.logger.info('creating an instance of ML_TubeWyrm')
        # Initialize TubeWyrm inheritance
        super().__init__(wait_sec=wait_sec, max_pulse_size=max_pulse_size, debug=debug)

        # model compatability checks
        if not isinstance(model, sbm.WaveformModel):
            emsg = f'Specified model-type {type(model)} is incompatable with this submodule'
            if self.debug:
                self.logger.critical(emsg)
            else:
                raise TypeError(emsg)
        else:
            self.model_name = model.name
            ml_kwargs.update({'model': model})
            if self.debug:
                self.logger.info(f'self.model_name "{self.model_name}" assigned')
        
        # weight_names compatability checks
        if not isinstance(weight_names, (str, list, tuple)):
            emsg = f'weight_names of type {type(weight_names)} not supported.'
            if self.debug:
                self.logger.critical(emsg)
            else:
                raise TypeError(emsg)
        elif isinstance(weight_names, str):
            weight_names = [weight_names]
        
        if not all(isinstance(_e, str) for _e in weight_names):
            emsg = f'not all elements of weight_names is type str - not supported'
            if self.debug:
                self.logger.critical(emsg)
            else:
                raise TypeError(emsg)
        # Cross checks with `model`
        else:
            passed_wn = []
            for _wn in weight_names:
                try:
                    model.from_pretrained(_wn)
                    passed_wn.append(_wn)
                    if self.debug:
                        self.logger.debug(f'weight "{_wn}" loaded successfully')
                except ValueError:
                    emsg = f'model weight "{_wn}" is incompatable with {model.name} - skipping weight'
                    if self.debug:
                        logging.warning(emsg)
                    else:
                        pass
            weight_names = passed_wn
            if len(passed_wn) == 0:
                emsg = 'No accepted weight names'
                if self.debug:
                    self.logger.critical(emsg)
                else:
                    raise ValueError(emsg)
            else:
                ml_kwargs.update({'weight_names': weight_names})
                if self.debug:
                    self.logger.info(f'self.weight_names assigned')


        # Ensure equivalent parameters match those in model
        # ww_kwargs homogenization
        if 'target_samprate' in ww_kwargs.keys(): 
            ww_kwargs.update({'target_samprate': model.sampling_rate})
        if 'target_overlap' in ww_kwargs.keys():
            ww_kwargs.update({'target_overlap': model._annotate_args['overlap'][1]})
        if 'target_blinding' in ww_kwargs.keys():
            ww_kwargs.update({'target_blinding': model._annotate_args['blinding'][1][0]})
        if 'target_order' in ww_kwargs.keys():
            ww_kwargs.update({'target_order': model.component_order})

        # ml_kwargs homogenization
        if 'max_samples' in ml_kwargs.keys():
            if ml_kwargs['max_samples'] < 2*model.in_samples:
                ml_kwargs.update({'max_samples': 2*model.in_samples})
        if 'weight_names' in ml_kwargs.keys():
            ml_kwargs.update({'weight_names': weight_names})
        
        # If executing with recursive debugging, overwrite debugging status
        # for component Wyrms' input kwargs
        if recursive_debug:
            ww_kwargs.update({'debug': self.debug})
            pw_kwargs.update({'debug': self.debug})
            ml_kwargs.update({'debug': self.debug})
        
        # Initialize Component Wyrms from kwarg dictionaries
        # windowwyrm
        wind_d = WindowWyrm(**ww_kwargs)
        self.logger.info('created component WindowWyrm')
        # procwyrm
        proc_d = ProcWyrm(**pw_kwargs)
        self.logger.info('created component ProcWyrm')
        # waveformmodelwyrm
        wfm_d = WaveformModelWyrm(**ml_kwargs)
        self.logger.info('created component WaveformModelWyrm')

        # Compose wyrm_dict using TubeWyrm's update class method
        self.update({'window': wind_d, 'process': proc_d, 'predict': wfm_d})
        self.logger.info('updated self.wyrm_dict with component wyrms')

