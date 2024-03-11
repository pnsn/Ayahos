import torch
import wyrm.util.compatability as wuc
import seisbench.models as sbm
import numpy as np
from collections import deque
from obspy import UTCDateTime, Trace
from wyrm.core.trace import MLTrace, MLTraceBuffer
from wyrm.core._base import Wyrm
from wyrm.core.dictstream import DictStream, ComponentStream


class WindowWyrm(Wyrm):
    """
    The WindowWyrm class takes windowing information from an input
    seisbench.models.WaveformModel object and user-defined component
    mapping and data completeness metrics and provides a pulse method
    that iterates across entries in an input DictStream object and
    generates windowed copies of data therein that pass data completeness
    requirements. Windowed data are formatted as as deque of DictStream 
    objects and
    staged in the WindWyrm.queue attribute (a deque) that can be accessed
    by subsequent data pre-processing and ML prediction Wyrms.
    """

    def __init__(
        self,
        ref_comp='Z',
        component_aliases={"Z": "Z3", "N": "N1", "E": "E2"},
        ref_comp_thresh=0.95,
        model_name="EQTransformer",
        target_sampling_rate=100.0,
        target_npts=6000,
        target_overlap=1800,
        max_pulse_size=20,
        debug=False,
    ):
        """
        Initialize a WindowWyrm object that samples a DictStream of MLTraceBuffer
        objects and generates ComponentStream copies of windowed data 

        :: INPUTS ::
        :param ref_comp: [str] reference component code passed to initialization
                    of ComponentStream objects
        :param component_aliases: [dict] dictionary defining aliases for component
                    codes with keys representing the aliases and values representing
                    iterable strings of component names that correspond to each alias
        :param ref_comp_thresh: [float] fractional completeness threshold value for the
                    reference component when generating a given ComponentStream
        :
        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        if not isinstance(component_aliases, dict):
            raise TypeError('component_aliases must be type dict')
        # elif not all(_c in 'ZNE' for _c in component_aliases.keys()):
        #     raise KeyError('component_aliases keys must comprise "Z", "N", "E"')
        elif not all(isinstance(_v, str) and _k in _v for _k, _v in component_aliases.items()):
            raise SyntaxError('component_aliases values must be type str and include the key value')
        else:
            self.component_aliases = component_aliases

        if not isinstance(ref_comp, str):
            raise TypeError
        elif ref_comp not in self.component_aliases.keys():
            raise KeyError
        else:
            self.ref_comp = ref_comp
        
        self.ref_comp_thresh = wuc.bounded_floatlike(
            ref_comp_thresh,
            name='ref_comp_thresh',
            minimum=0,
            maximum=1,
            inclusive=True
        )

        if not isinstance(model_name, str):
            raise TypeError('model_name must be type str')
        else:
            self.model_name = model_name

        # Target sampling rate compat check
        self.target_sampling_rate = wuc.bounded_floatlike(
            target_sampling_rate,
            name='target_sampling_rate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target window number of samples compat check
        self.target_npts = wuc.bounded_intlike(
            target_npts,
            name='target_npts',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # Target window number of sample overlap compat check
        self.target_overlap = wuc.bounded_intlike(
            target_overlap,
            name='target_overlap',
            minimum=0,
            maximum=None,
            inclusive=False
        )

         # Set Defaults and Derived Attributes
        # Calculate window length, window advance, and blinding size in seconds
        self.window_sec = self.target_npts/self.target_sampling_rate
        self.advance_sec = (self.target_npts - self.target_overlap)/self.target_sampling_rate

        # Create dict for holding instrument window starttime values
        self.default_starttime = UTCDateTime('2599-12-31T23:59:59.999999')
        self.window_tracker = {}

        # Create queue for output collection of windows
        self.queue = deque([])


    def pulse(self, x):
        if not isinstance(x, DictStream):
            raise TypeError
        for _ in range(self.max_pulse_size):
            self.update_window_tracker(x)
            nnew = self.sample_windows(x)
            if nnew == 0:
                break
        y = self.queue
        return y

    def update_window_tracker(self, dst):
        # Split by site
        dict_x_site = dst.split_on_key(key='site')
        # Iterate across site holdings
        for site, dst_site in dict_x_site.keys():
            # if site is not in window_tracker, get max_starttime from a ref_comp DictStream
            if site not in self.window_tracker.keys():
                dst_ref = dst_site.fnselect(f'*.*.*.*[{self.component_aliases[self.ref_comp]}].*.*')
                if len(dst_ref) > 0:
                    init_t0 = dst_ref.stats.max_starttime
                    self.window_tracker.update({site: {'init t0': init_t0}})
            else:
                init_t0 = self.window_tracker[site]['init t0']
            # Iterate across instruments at site
            dst_site_dict_x_inst = dst_site.split_on_key(key='inst')
            for inst, dst_inst in dst_site_dict_x_inst.keys():
                # If instrument is not in window_tracker, get init_t0 for starttime + windowing instructions
                if inst not in self.window_tracker[site].keys():
                    # If init_t0 is in the window, use init_t0 as the window reference time
                    if dst_inst.max_starttime < init_t0 < dst_inst.min_endtime:
                        self.window_tracker[site].update({inst: init_t0})
                    # If init_t0 is before the start of the window, increment up
                    elif dst_inst.

        




    def update_window_tracker(self, x):
        """
        Iterate across sites/instruments in DictStream "x" and
        update the window_tracker attribute with the following potential behaviors
        1) Create new site/instrument branch in window_tracker for undocumented
            site/instruments in "x"
        2) 
        """
        if not isinstance(x, DictStream):
            raise TypeError
        ### DO CHECK FOR WINDOW INDEXING ###
        for site in x.site_inst_index.keys():
            # If site is not present in the window_index
            if site not in self.window_tracker.keys():
                self.window_tracker.update({site: {}})
                # Filter down to reference trace(s)
                _xsite = x.fnselect(f'{site}.??.??{self.component_aliases[self.ref_comp]}*')
                # Iterate across instruments at this site and add them to the list
                for inst in _xsite.site_inst_index[site].keys():
                    # Use DictStream header attribute 'max_starttime' to get starttime for windowing
                    self.window_tracker[site].update({inst: _xsite.max_starttime})
                # tag the initialization t0 used (perhaps redundant, but useful...?)
                self.window_tracker[site].update({'init t0': _xsite.max_starttime})

            # If site is present in the window_index
            else:
                # Iterate across instrument codes
                for inst in self.site_inst_index[site]:
                    # If there is a new instrument
                    if inst not in self.window_tracker[site]:
                        # Get the initiation time
                        init_t0 = self.window_tracker[site]['init t0']
                        # Get instrument slice of the DictStream
                        _xinst = x.fnselect(f'{site}.{inst}{self.component_aliases[self.ref_comp]}*')
                        # Increment up window 
                        dt_adv = _xinst.min_starttime - init_t0
                        nadv = dt_adv // self.advance_sec
                        # Add one additional advance to start within the data feed
                        # to avoid a front gap
                        first_t0 = init_t0 + (nadv + 1)*self.advance_sec
                        # Set initial insturment timing to the init time
                        self.window_tracker[site].update({inst: first_t0})

                    # If the instrument is in the window_tracker
                    else:
                        # assess if the current window starttime can still make a 
                        # viable window given the buffer timings
                        wts = self.window_tracker[site][inst]
                        wte = wts + self.window_sec
                        # If the window end-time is earlier than the earliest starttime
                        if wte < _xinst.stats.min_starttime:
                            # Increment up window index
                            dt_adv = _xinst.stats.min_starttime - wte
                            nadv = dt_adv // self.advance_sec
                            self.window_tracker[site][inst] += (nadv + 1)*self.advance_sec
        # END
            
    def sample_windows(self, x):
        """
        Using the window starttimes contained in self.window_tracker for each instrument,
        assess if there is enough data on a given reference channel for that instrument
        to generate a windowed copy of data contained in the MLTraceBuffers for the
        instrument. Windowed data are copied as a DictStream and they have the following
        header data (DictStream.stats) updated:
            stats.ref_model
            stats.ref_starttime
            stats.ref_sampling_rate
            stats.ref_npts

        This class method heavily leverages attributes and methods from the
        DictStream class.

        :: INPUT ::
        :param x: [wyrm.core.data.dictstream.DictStream] 
                    Dictionary Stream from which data will be sampled
                    class methods applied:
                        apply_trace_method()
                        assess_window_readiness()



        :: CONTRIBUTING ATTRIBUTES ::
        :attr window_tracker: [dict] tiered dictionary with structure:
                                    {site_code: {
                                        instrument_code: 
                                            window_starttime}}
        :attr queue: [collections.deque] appends a copy of windowed data in
                        a DictStream object
        :attr comp_map: [dict] dictionary of aliases for component codes that
                        correspond to standardized component codes Z, N, E
        :attr ref_comp: [str] reference component code
        :attr
        """
        if not isinstance(x, DictStream):
            raise TypeError
        nnew = 0
        # Split view into heirarchical structure (NOTE - JUST REINDEXING OF ORIGINAL DATA)
        x_hdict = x.split_by_instrument(heirarchical=True)
        # Iterate across window_tracker entries
        for site, values in self.window_tracker.items():
            for inst, wts in values.items():
                # Get a copy of the instrument specific DictStream (MAKES COPIES HERE)
                _dst = x_hdict[site][inst].copy()
                # Populate window time band
                wte = wts + self.window_sec
                # Trim copy to specified window band
                _dst.trim(starttime=wts, endtime=wte, pad=True, fill_value=None)
                # If the DictStream passes windowing requirements
                if self.is_dictstream_window_viable(_dst):
                    # Update DictStream stats information
                    _dst.stats.ref_starttime = wts
                    _dst.stats.ref_sampling_rate = self.target_sampling_rate
                    _dst.stats.ref_npts = self.target_npts
                    _dst.stats.ref_model = self.model_name
                    # Run an initial sync check
                    _dst.is_syncd(run_assessment=True)
                    # Append windowed data to queue
                    self.queue.append(_dst)
                    # Advance window timing
                    self.window_tracker[site][inst] += self.advance_sec
                    # Increment nnew
                    nnew += 1
        return nnew
            
    def update_windowing_params_from_seisbench(self, model):
        """
        Helper method for (re)setting the window-defining attributes for this
        WindowWyrm object from a seisbench.models.WaveformModel object:

            self.model_name = model.name
            self.sampling_rate = model.sampling_rate
            self.target_npts = model.in_samples
            self.target_overlap = model._annotate_args['overlap']

        """
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError
        elif model.name != 'WaveformModel':
            if model.sampling_rate is not None:
                self.target_sampling_rate = model.sampling_rate
            if model.in_samples is not None:
                self.target_npts = model.in_samples
            self.target_overlap = model._annotate_args['overlap'][1]
            self.model_name = model.name
        else:
            raise TypeError('seisbench.models.WaveformModel base class does not provide the necessary update information')



class ProcessWyrm(Wyrm):
    """
    A submodule for applying a class method with specified key-word arguments
    """
    def __init__(
        self,
        max_pulse_size=10000,
        debug=False,
        target_class=ComponentStream,
        method_name='filter',
        mkwargs={'type': 'bandpass',
                 'freqmin': 1,
                 'freqmax': 45}
        ):

        super().__init__(max_pulse_size=max_pulse_size, debug=debug)




        if isinstance(method_name, str):
            self.method=method_name
        else:
            raise TypeError
        if isinstance(dict, )
        self.kwargs = method_kwargs
    
    def pulse(self, x):




class PreProcessWyrm(Wyrm):
    
    def __init__(
        self,
        channel_fill_rule='zeros',
        completeness_threshold=0.8,
        comp_map = {'Z': 'Z3', 'N': 'N1', 'E': 'E2'}
        ref_comp = 'Z'
        order='ZNE',
        gap_filling_kwargs={},
        norm_type='max',
        max_pulse_size=10000,
        debug=False
    ):
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        if norm_type not in ['max','std']:
            raise ValueError
        else:
            self.norm_type = norm_type

    def pulse(self, x):
        if not isinstance(x, deque):
            raise TypeError
        qlen = len(x)
        for _i in range(self.max_pulse_size):
            if qlen == 0:
                break
            elif _i + 1 > qlen:
                break
            else:
                _x = x.popleft()

            if not isinstance(_x, DictStream):
                x.append(_x)
            else:
                _x.
            
        


    # def apply_fill_rule(self, dictstream):
    #     dictstream.apply_fill_rule(ref_code=self.ref_comp, )



    #     elif self.channel_fill_rule == 'clonez':
    #         self._apply_clonez()
    #     elif self.channel_fill_rule == 'c'


class PredictionWyrm(Wyrm):
    """
    Conduct ML model predictions on preprocessed data ingested as a deque of
    ComponentStream objects using one or more pretrained model weights. Following
    guidance on model application acceleration from SeisBench, an option to precompile
    models on the target device is included as a default option.    
    """
    def __init__(
        self,
        model,
        weight_names,
        devicetype='cpu',
        compiled=True,
        max_pulse_size=1000,
        debug=False,
        **options):
        """
        Initialize a PredictionWyrm object
        """
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        
        # model compatability checks
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be a seisbench.models.WaveformModel object')
        elif model.name == 'WaveformModel':
            raise TypeError('model must be a child-class of the seisbench.models.WaveformModel class')
        else:
            self.model = model
        
        if 'blinding' in options.keys():
            self.blinding = wuc.bounded_intlike(
                options['blinding'],
                name = 'blinding',
                minimum=0,
                maximum=model.in_samples//2,
                inclusive=True
            )
        else:
            self.blinding = model._annotate_args['blinding'][1][0]
        
        if 'stacking' in options.keys():
            if options['stacking'] in ['max','avg']:
                self.stacking_method=options['stacking']
        else:
            self.stacking_method = model._annotate_args['stacking'][1]
        

        # Model weight_names compatability checks
        pretrained_list = model.list_pretrained()
        if isinstance(weight_names, str):
            weight_names = [weight_names]
        elif isinstance(weight_names, (list, tuple)):
            if not all(isinstance(_n, str) for _n in weight_names):
                raise TypeError('not all listed weight_names are type str')
        else:
            for _n in weight_names:
                if _n not in pretrained_list:
                    raise ValueError(f'weight_name {_n} is not a valid pretrained model weight_name for {model}')
        self.weight_names = weight_names

        # device compatability checks
        if not isinstance(devicetype, str):
            raise TypeError('devicetype must be type str')
        else:
            try:
                device = torch.device(devicetype)
            except RuntimeError:
                raise RuntimeError(f'devicetype {devicetype} is an invalid device string for PyTorch')
            try:
                self.model.to(device)
            except RuntimeError:
                raise RuntimeError(f'device type {devicetype} is unavailable on this installation')
            self.device = device
        
        # Preload/precompile models
        if isinstance(compiled, bool):    
            self.compiled = compiled
        else:
            raise TypeError(f'"compiled" type {type(compiled)} not supported. Must be type bool')

        self.cmods = {}
        for wname in self.weight_names:
            if self.debug:
                print(f'Loading {self.model.name} - {wname}')
            cmod = self.model.from_pretrained(wname)
            if compiled:
                if self.debug:
                    print(f'...pre compiling model on device type "{self.device.type}"')
                cmod = torch.compile(cmod.to(self.device))

            self.cmods.update({wname: cmod})



    def pulse(self, x):
        if not isinstance(x, deque):
            raise TypeError('input "x" must be type deque')
        
        qlen = len(x)
        batch_data = []
        batch_meta = []

        for _i in range(self.max_pulse_size):
            if len(x) == 0:
                break
            if _i == qlen:
                break
            else:
                _x = x.pop()
                if not(isinstance(_x, DictStream)):
                    x.appendleft(_x)
                else:
                    try:
                        _data = _x.to_numpy_array()
                        _meta = _x.stats
                    except ValueError:
                        x.appendleft(_x)
                        continue
                    batch_data.append(torch.Tensor(_data))
                    batch_meta.append(_meta)

        batch_data = torch.concat(batch_data)

        for wname, model in self.cmods.items():
            batch_pred = self.run_prediction(model, batch_data, batch_meta)
            self.batch2queue(model, wname, batch_pred, batch_meta)
            del batch_pred
        del batch_data
        y = self.queue
        return y

    def run_prediction(self, model, batch_data, batch_meta):
        """
        Run a ML prediction on an input batch of windowed data using self.model on
        self.device. Provides checks that self.model and batch_data are appropriately
        formatted and staged on the same device

        :: INPUT ::
        :param batch_data: [numpy.ndarray] or [torch.Tensor] data array with scaling
                        appropriate to the input layer of self.model
        :: OUTPUT ::
        :return batch_preds: [torch.Tensor] or [tuple] thereof - prediction outputs
                        from self.model
        """
        if not isinstance(batch_data, (torch.Tensor, np.ndarray)):
            raise TypeError('batch_data must be type torch.Tensor or numpy.ndarray')
        elif isinstance(batch_data, np.ndarray):
            batch_data = torch.Tensor(batch_data)
        
        if model.device.type != self.device.type and not self.compiled:
            model.to(self.device)
        
        if batch_data.device.type != self.device.type:
            batch_preds = model(batch_data.to(self.device))
        else:
            batch_preds = model(batch_data)
        
        out_shape = (len(batch_meta), len(model.labels), model.in_samples)
        if isinstance(batch_preds, (tuple, list)):
            if all(isinstance(_p, torch.Tensor) for _p in batch_preds):
                batch_preds = torch.concat(batch_preds)
            else:
                raise TypeError('not all elements of preds is type torch.Tensor')
        
        if batch_preds.shape != out_shape:
            batch_preds.reshape(out_shape)

        return batch_preds

    def batch2queue(self, model, weight_name, batch_preds, batch_meta):
        # Detach prediction array and convert to numpy
        if batch_preds.device.type != 'cpu':
            batch_preds = batch_preds.detach().cpu().numpy()
        else:
            batch_preds = batch_preds.detach().numpy()
        # Iterate across metadata dictionaries
        for _i, _meta in batch_meta:
            # Iterate across prediction labels
            for _l, label in enumerate(model.labels):
                # Compose output trace from prediction values and header data
                _mlt = MLTrace(data = batch_preds[_i, _l, :], header=_meta)
                # Update component labeling
                _mlt.set_component(new_label=label)
                # Update weight name in mltrace header
                _mlt.stats.weight = weight_name
                # Add mltrace to dsbuffer (subsequent buffering to happen in the next step)
                self.queue.append(_mlt)









        #     # Grab site/instrument dictstream objects
        #     site_ready = []
        #     for sidst in x_hdict[site].values():
        #         # Create a trimmed copy for the candidate window
        #         _sidst = sidst.copy().apply_trace_method(
        #             'trim',
        #             starttime=t0_ref,
        #             endtime=t0_ref + self.window_sec,
        #             pad=True, 
        #             fill_value=None)
        #         proc_string = f'Wyrm 0.0.0: apply_trace_method("trim", starttime={t0_ref}, '
        #         proc_string += f'endtime={t0_ref + self.window_sec}, pad=True, fill_value=None)'
        #         # Assess if the reference trace 
        #         ready =  _sidst.assess_window_readiness(
        #             ref_comp=self.ref_comp,
        #             ref_comp_thresh=self.ref_comp_thresh,
        #             comp_map = self.comp_map)
        #         if ready:
        #             # Update DictStream header with windowing attributes
        #             _sidst.stats.ref_starttime = t0_ref
        #             _sidst.stats.ref_sampling_rate = self.target_sampling_rate
        #             _sidst.stats.ref_npts = self.target_npts
        #             _sidst.stats.ref_model = self.model_name
        #             # Run assessment for trace sync
        #             _sidst.is_syncd()
        #             # append processing string to DictStream header (trim)
        #             _sidst.stats.processing.append(proc_string)
        #             # append processing string to DictStream header (passed ref_comp_thresh)
        #             proc_string = f'Wyrm 0.0.0: PASS: assess_window_readiness(ref_comp={self.ref_comp}), '
        #             proc_string += f'ref_comp_thresh={self.ref_comp_thresh}, comp_map={self.comp_map})'
        #             _sidst.stats.processing.append(proc_string)
                    
        #             # Append to WindowWyrm queue    
        #             self.queue.append(_sidst)
        #             # Update nnew
        #             nnew += 1
        #             site_ready.append(True)
        #         else:
        #             site_ready.append(False)
        #     # If any instruments produced a window for a site, increment window_tracker[site]
        #     if any(site_ready):
        #         self.window_tracker[site] += self.advance_sec

        # return nnew