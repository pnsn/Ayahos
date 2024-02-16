import seisbench.models as sbm
import wyrm.util.input_compatability_checks as icc


def get_overlap(model):
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    return model._annotate_args["overlap"][-1]


def get_blinding(model):
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    return model._annotate_args["blinding"][-1]


def get_stacking_method(model):
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    return model._annotate_args["stacking"][-1]


def change_overlap(model, new_overlap=500):
    """
    Change the number of prediction samples that successive
    windows for prediction overlap

    :: INPUTS ::
    :param model: [seisbench.model.WaveformModel] model to update
    :param new_overlap: [int] new number of samples for overlap
                        must be a positive integer smaller than
                        model.in_samples
    :: OUTPUT ::
    :param model: [seisbench.model.WaveformModel] model object with updated
                overlap values in model._annotate_args['overlap']
    """
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    new_overlap = icc.bounded_intlike(
        new_overlap,
        name="new_overlap",
        minimum=0,
        maximum=model.in_samples,
        inclusive=False,
    )
    model._annotate_args["overlap"][-1] = new_overlap
    return model


def change_blinding(model, new_blinding=200):
    """
    Change the number of prediction samples to discard
    on each side of a prediction window for input model

    :: INPUTS ::
    :param model: [seisbench.model.WaveformModel] model to update
    :param new_blinding: [int] new number of samples for blinding
                        must be a positive integer smaller than half
                        the model.in_samples
    :: OUTPUT ::
    :param model: [seisbench.model.WaveformModel] model object with updated
                blinding values in model._annotate_args['blinding']
    """
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    new_blinding = icc.bounded_intlike(
        new_blinding,
        name="new_blinding",
        minimum=0,
        maximum=model.in_samples // 2,
        inclusive=True,
    )
    model._annotate_args["blind"][-1] = (new_blinding, new_blinding)
    return model


def change_sampling_rate(model, new_sampling_rate=200.0):
    """
    Change the sampling rate expected by a seisbench.model.WaveformModel
    object (model.sampling_rate)
    """
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    new_sampling_rate = icc.bounded_floatlike(
        new_sampling_rate,
        name="new_sampling_rate",
        minimum=0,
        maximum=1e12,
        inclusive=False,
    )
    model.sampling_rate = new_sampling_rate
    return model


def change_norm_method(model, new_norm_method):
    """
    Change the string designation for a seisbench.models.WaveformModel
    object if it has the attribute model.norm
    """
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError("model must be type seisbench.models.WaveformModel")
    if not isinstance(new_norm_method, str):
        raise TypeError("new_norm_method must be type str")
    if new_norm_method.lower() in ["minmax", "std", "peak"]:
        try:
            model.norm = new_norm_method.lower()
        except AttributeError:
            raise AttributeError(f"model {model.name} does not have a norm attribute")
    else:
        raise ValueError(
            f'new_norm_method "{new_norm_method}" not supported. Use "minmax", "peak", "std"'
        )
    return model


def change_stacking_method(model, new_stacking_method):
    """
    Change the string designator for prediction stacking
    :: INPUTS ::

    """
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError('model must be type seisbench.models.WaveformModel')
    if not isinstance(new_stacking_method, str):
        raise TypeError("new_stacking_method must be type str")
    if new_stacking_method.lower() in ["max", "avg"]:
        model._annotate_args["stacking"][-1] = new_stacking_method.lower()
    else:
        raise ValueError(
            f'new_stacking_method "{new_stacking_method}" not supported. Use "max" or "avg"'
        )


def change_model_windowing_params(
    model, norm=None, sampling_rate=None, blinding=None, overlap=None, stacking=None
):
    """
    Wrapper method for the collection of model windowing parameter
    change methods in this module. See individual methods for input
    requirements. Default value for each is None which skips a given
    change method

    :: INPUTS ::
    :param model: [seisbench.models.WaveformModel] model to alter
    :param norm:           change_norm_method()
    :param sampling_rate:  change_sampling_rate()
    :param blinding:       change_blinding()
    :param overlap:        change_overlap()
    :param stacking:       change_stacking_method()

    :: OUTPUT ::
    :return model: [seisbench.models.WaveformModel] model with altered
                    windowding parameters
    """
    if not isinstance(model, sbm.WaveformModel):
        raise TypeError('model must be type seisbench.models.WaveformModel')
    if norm is not None:
        model = change_norm_method(model, new_norm_method=norm)
    if sampling_rate is not None:
        model = change_sampling_rate(model, new_sampling_rate=sampling_rate)
    if blinding is not None:
        model = change_blinding(model, new_blinding=blinding)
    if overlap is not None:
        model = change_overlap(model, new_overlap=overlap)
    if stacking is not None:
        model = change_stacking_method(model, new_stacking_method=stacking)
    return model
