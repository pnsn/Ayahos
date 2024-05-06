import seisbench.models as sbm
from wyrm.util.input import bounded_intlike


def pretrained_dict():
    out = {'EQTransformer': ['ethz',
                             'geofon',
                             'instance',
                             'iquique',
                             'lendb',
                             'neic',
                             'obs',
                             'original',
                             'original_nonconservative',
                             'pnw',
                             'scedc',
                             'stead'],
           'PhaseNet': ['diting',
                        'ethz',
                        'geofon',
                        'instance',
                        'iquique',
                        'lendb',
                        'neic',
                        'obs',
                        'original',
                        'scedc',
                        'stead']}
    return out

def update_windowing_params(model, blinding=False, overlap=False):
    """
    Update the model._annotate_args 'blinding' and/or 'overlap' values
    for a seisbench.models.WaveformModel object

    :: INPUTS ::
    :param model: [seisbench.models.WaveformModel] model to modify
    :param blinding: [int-like] non-negative number of samples to blind on either
                    end of prediction windows
                    [False] - do not modify 
    :param overlap: [int-like] non-negative number of samples sequential windows
                    should overlap by for this model
                    [False] - do not modify 
    
    NOTE: if overlap < blinding, this will raise a UserWarning because this
            configuration will result in data sampling gaps.

    :: OUTPUT ::
    :return model: [seisbench.models.WaveformModel] modified model
    """


    if not isinstance(model, sbm.WaveformModel):
        raise TypeError
    # modify blinding if a value is passed
    if blinding:
        blinding = bounded_intlike(
            blinding,
            name='blinding',
            minimum=0,
            maximum=0.5*model.in_samples,
            inclusive=True)

        model._annotate_args['blinding'][1] = (blinding, blinding)
    
    else:
        blinding = model._annotate_args['blinding'][1]
    
    # modify overlap if a value is passed
    if overlap:
        overlap = bounded_intlike(
            overlap,
            name='overlap',
            minimum=-1,
            maximum=model.in_samples,
            inclusive=False
        )
        model._annotate_args['overlap'] = (
            model._annotate_args['overlap'][0],
            overlap
        )
    else:
        overlap = model._annotate_args['overlap'][1]
    
    # Run sanity check on blinding/overlap combination
    if overlap < blinding[0] or overlap < blinding[1]:
        raise UserWarning('Warning! Updated overlap/blinding will result in gaps!')
    else:
        return model