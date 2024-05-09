import seisbench.models as sbm
from wyrm import *

def compose_mldetect_tube(
        model=sbm.EQTransformer(),
        weight_names=['pnw'],
        devicetype='cpu',
        norm_type='peak',
        missing_channel_rule='zeros'
    ):
    """Compose a TubeWyrm object containing the necessary submodules
    to preprocess raw waveforms and conduct a ML detection/classification
    workflow for a specified SeisBench WaveformModel and associated
    pretrained weights

    :param model: ML model architecture with which to conduct predictions,
            defaults to sbm.EQTransformer()
    :type model: seisbench.models.WaveformModel child class
    :param weight_names: List of model.from_pretrained() compliant pretrained
            model weight names, defaults to ['pnw']
    :type weight_names: list of str
    :param devicetype: device name for PyTorch to run predictions on, defaults to 'cpu'
    :type devicetype: str, optional
        Supported option examples (also see torch.device)
            'cpu' - central processing unit
            'mps' - metal performance shaders (CPU + GPU on Apple Silicon)
            'gpu' - graphics processing unit
    :param norm_type: Type of data normalization to apply, defaults to 'peak'
    :type norm_type: str, optional
        Supported options:
            'peak'
            'std'
    :param missing_channel_rule: Missing channel fill rule name to apply, defaults to 'zeros'
    :type missing_channel_rule: str, optional
        Other supported options
            'clone_ref' - if non-reference channels (horizontal components) are missing or
                    too gappy, replace them with a clone of the reference channel (vertical)
            'clone_other' - if one non-reference channel is missing, clone the present 
                    non-reference channel to fill the missing component (i.e., clone one
                    horizontal)
    
    :: OUTPUT ::
    :return tube_wyrm: TubeWyrm object containing the composed processing sequence of Wyrm objects
    :rtype: wyrm.coordinating.sequencing.TubeWyrm

    """    

    # Initialize a WindowWyrm object with defaults
    windwyrm = WindowWyrm()
    # Update parameters using the seisbench model's attributes
    windwyrm.update_from_seisbench(model=model)

    # Initialize MethodWyrm objects for preprocessing
    # For treating gappy data
    mwyrm_gaps = MethodWyrm(
        pclass=WindowStream,
        pmethod='treat_gaps',
        pkwargs={})

    # For synchronizing temporal sampling and windowing
    mwyrm_sync = MethodWyrm(
        pclass=WindowStream,
        pmethod='sync_to_reference',
        pkwargs={'fill_value': 0})

    # For filling data out to 3-C from non-3-C data (either missing channels, or 1C instruments)
    mwyrm_fill = MethodWyrm(
        pclass=WindowStream,
        pmethod='apply_fill_rule',
        pkwargs={'rule': missing_channel_rule})

    # Initialize model specific normalization MethodWyrms
    mwyrm_norm = MethodWyrm(
        pclass=WindowStream,
        pmethod='normalize_traces',
        pkwargs={'norm_type': norm_type}
    )

    # Initialize MLDetectWyrm object for prediction work
    mlwyrm = MLDetectWyrm(model=model,
                          weight_names=weight_names,
                          devicetype=devicetype,
                          max_pulse_size=model._annotate_args['batch_size'][1])

    wyrm_dict = {'window': windwyrm,
                'treat_gaps': mwyrm_gaps,
                'sync_traces': mwyrm_sync,
                'channel_fill': mwyrm_fill,
                'normalize': mwyrm_norm,
                'predict': mlwyrm}


    # Finally compose tubewyrm
    tubewyrm = TubeWyrm(wyrm_dict=wyrm_dict,
                        max_pulse_size=1)
    return tubewyrm

if __name__ == '__main__':
    # Load model architecture from SeisBench
    model = sbm.EQTransformer()
    # Specify weights to include for this model
    weight_names = ['pnw','instance','stead','iquique','lendb']
    # Compose mltubewyrm
    mltubewyrm = compose_mldetect_tube(model=model, weight_names=weight_names)


