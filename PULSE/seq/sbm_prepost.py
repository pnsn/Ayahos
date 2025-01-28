from seisbench.models import sbm

from PULSE.seq.sequence import Sequence
from PULSE.data.dictstream import DictStream
from PULSE.mod.sampling import SamplingMod, WindowingMod
from PULSE.mod.buffering import BufferMod
from PULSE.mod.processing import ProcMod

def validate_sbm(model):
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError('model must be type seisbench.models.WaveformModel')
    elif model.name == 'WaveformModel':
        raise TypeError('The seisbench.models.WaveformModel baseclass does not provide a functional model')
    else:
        pass


def init_sbm_preprocessing_seq(model=sbm.EQTransformer(), run_validation=False, **ppkwargs):
    """Initialize a :class:`~PULSE.seq.sequence.Sequence` object containing
    a two-element sequence for windowing and preprocessing waveform data
    for input to a :class:`~seisbench.models.WaveformModel`-type PyTorch
    model architecture. 

    Sequence Comprises:
     - WindowingMod - parameterized with the input SeisBench model
                    **model** using :meth:`~PULSE.mod.sampling.WindowingMod.update_from_seisbench`
                    Generates :class:`~PULSE.data.window.Window` objects passed
                    to the subsequent module
     - ProcMod - preprocessing of :class:`~PULSE.data.window.Window` objects
                    using mostly default values

    :param model: _description_, defaults to sbm.EQTransformer()
    :type model: _type_, optional
    :param trace_fill_rule: _description_, defaults to 'primary'
    :type trace_fill_rule: str, optional
    :raises TypeError: _description_
    :raises TypeError: _description_
    :return: _description_
    :rtype: _type_
    """
    # Give option to run validation during INIT
    if run_validation:
        validate_sbm(model)

    # Initialize windmod using seisbench model parameterization helper metho
    windmod = WindowingMod().update_from_seisbench(model=model)
    # Initialize ppmod
    ppmod = ProcMod(pclass='PULSE.data.window.Window',
                         pmethod='preprocess',
                         mode='inplace',
                         pkwargs=ppkwargs)
    seq = Sequence([windmod, ppmod])

    return seq


def init_sbm_postprocessing_seq(
        model=sbm.EQTransformer(),
        run_validation=False,
        labels_to_buffer='PS',
        windows_to_buffer=12.,
        windows_to_lag_triggering = 1.,
        trigger_representation='simple_max',
        pick_format='obspy'):
    """Initialize a :class:`~PULSE.seq.sequence.Sequence` object
    that provides a post-processing module sequence for outputs
    from a :class:`~PULSE.mod.detecting.SBMMod` module largely using
    metadata carried by **model** to parameterize these

    """    
    if run_validation:
        validate_sbm(model)

    wlen = (model.in_samples - 1)/model.sampling_rate
    olen = model._annotate_args['overlap'][1]/model.sampling_rate
    slen = wlen - olen
    max_buffer_len = wlen + (windows_to_buffer - 1)*slen
    trigger_delay = wlen + (windows_to_lag_triggering - 1)*slen

    # Initialize label selector
    labelfiltermod = ProcMod(
        pclass='PULSE.data.dictstream.DictStream',
        mode='output',
        pmethod='select',
        pkwargs={'component': f'[{labels_to_buffer}]'})
    # Initizalize blinding 
    blindmod = ProcMod(
        pclass='PULSE.data.dictstream.DictStream',
        mod='inplace',
        pmethod='blind',
        pkwargs={'npts':model._annotate_args['blinding'][1]}
        )
    # Initialize prediction buffering
    stackmod = BufferMod(
        method=model._annotate_args['stacking'][1],
        fill_value=0.,
        maxlen=max_buffer_len)
    # Initialize prediction triggering
    postsamplingmod = SamplingMod(
        ref_val=labels_to_buffer[0],
        blind_after_sampling=True,

                             )
    
