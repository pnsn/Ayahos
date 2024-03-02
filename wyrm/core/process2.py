from collections import deque
from obspy import UTCDateTime, Trace
from wyrm.core.trace import MLTrace, MLTraceBuffer
from wyrm.core._base import Wyrm
from wyrm.core.dictstream import DictStream
import wyrm.util.compatability as wuc
import seisbench.models as sbm


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
        code_map={"Z": "Z3", "N": "N1", "E": "E2"},
        ref_comp_thresh=0.95,
        model_name="EQTransformer",
        target_sampling_rate=100.0,
        target_npts=6000,
        target_overlap=1800,
        max_pulse_size=20,
        debug=False,
    ):
        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        if not isinstance(code_map, dict):
            raise TypeError('code_map must be type dict')
        elif not all(_c in 'ZNE' for _c in code_map.keys()):
            raise KeyError('code_map keys must comprise "Z", "N", "E"')
        elif not all(isinstance(_v, str) and _k in _v for _k, _v in code_map.items()):
            raise SyntaxError('code_map values must be type str and include the key value')
        else:
            self.code_map = code_map

        if not isinstance(ref_comp, str):
            raise TypeError
        elif ref_comp not in self.code_map.keys():
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
            self.check_for_new_data_feeds(x)
            nnew = self.sample_windows(x)
            if nnew == 0:
                break
        y = self.queue
        return y

    def check_for_new_data_feeds(self, x):
        """
        Check for new station data feeds with reference channels,
        and if new ones are found, add them to the window_tracker
        attribute and scrape the reference traces for instrument(s)
        at each site for the earliest
        """
        if not isinstance(x, DictStream):
            raise TypeError
        ### DO CHECK FOR WINDOW INDEXING ###
        # Filter for reference component records (does create copy)
        x_ref = x.fnselect(f'*.*.??[{self.comp_map[self.ref_comp]}].*')
        # Split data into site-keyed dictstreams
        x_ref_sites = x_ref.split_by_site()
        # Iterate across site-level split of dictstream
        for site_code, sdst in x_ref_sites.items():
            # If this is a new site code
            if site_code not in self.window_tracker.keys():
                # Scrape the reference data for a first_t0 value
                first_t0 = self.default_t0 
                for _tr in sdst.values():
                    if _tr.stats.starttime < first_t0:
                        first_t0 = _tr.stats.starttime
                # and update
                self.window_tracker.update({site_code: first_t0})
            # If this is an existing site code
            if self.window_tracker[site_code] < 
        # END FOR LOOP
    
    def sample_windows(self, x):
        if not isinstance(x, DictStream):
            raise TypeError
        nnew = 0
        # Split view into heirarchical structure (NOTE - JUST REINDEXING OF ORIGINAL DATA)
        x_hdict = x.split_by_instrument(heirarchical=True)
        # Iterate across window_tracker entries
        for site, t0_ref in self.window_tracker.items():
            # Grab site/instrument dictstream objects
            site_ready = []
            for sidst in x_hdict[site].values():
                # Create a trimmed copy for the candidate window
                _sidst = sidst.copy().apply_trace_method(
                    'trim',
                    starttime=t0_ref,
                    endtime=t0_ref + self.window_sec,
                    pad=True, 
                    fill_value=None)
                proc_string = f'Wyrm 0.0.0: apply_trace_method("trim", starttime={t0_ref}, '
                proc_string += f'endtime={t0_ref + self.window_sec}, pad=True, fill_value=None)'
                # Assess if the reference trace 
                ready =  _sidst.assess_window_readiness(
                    ref_comp=self.ref_comp,
                    ref_comp_thresh=self.ref_comp_thresh,
                    comp_map = self.comp_map)
                if ready:
                    # Update DictStream header with windowing attributes
                    _sidst.stats.ref_starttime = t0_ref
                    _sidst.stats.ref_sampling_rate = self.target_sampling_rate
                    _sidst.stats.ref_npts = self.target_npts
                    _sidst.stats.ref_model = self.model_name
                    # Run assessment for trace sync
                    _sidst.is_syncd()
                    # append processing string to DictStream header (trim)
                    _sidst.stats.processing.append(proc_string)
                    # append processing string to DictStream header (passed ref_comp_thresh)
                    proc_string = f'Wyrm 0.0.0: PASS: assess_window_readiness(ref_comp={self.ref_comp}), '
                    proc_string += f'ref_comp_thresh={self.ref_comp_thresh}, comp_map={self.comp_map})'
                    _sidst.stats.processing.append(proc_string)
                    
                    # Append to WindowWyrm queue    
                    self.queue.append(_sidst)
                    # Update nnew
                    nnew += 1
                    site_ready.append(True)
                else:
                    site_ready.append(False)
            # If any instruments produced a window for a site, increment window_tracker[site]
            if any(site_ready):
                self.window_tracker[site] += self.advance_sec

        return nnew
            
    def update_windowing_params_from_seisbench(self, model):
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError
        elif model.name != 'WaveformModel':
            if model.sampling_rate is not None:
                self.target_sampling_rate = model.sampling_rate
            if model.in_samples is not None:
                self.target_npts = model.in_samples
            self.target_overlap = model._annotate_args['overlap'][1]
        else:
            raise TypeError('seisbench.models.WaveformModel base class does not provide the necessary update information')