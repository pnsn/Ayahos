"""
Module for handling Ayahos :class: `~ayahos.submodules.sbmtubewyrm.SBMTubeWyrm` objects

:author: Nathan T. Stevens
:org: Pacific Northwest Seismic Network
:email: ntsteven (at) uw.edu
:license: AGPL-3.0
"""

import seisbench.models as sbm
from ayahos.wyrms import *
from ayahos import WindowStream

class SBMTubeWyrm(TubeWyrm):

    def __init__(
            self,
            model,
            weight_names,
            devicetype='cpu',
            fill_rule='zeros',
            min_batch_size=1,
            max_pulse_size=1
    ):
        """Initialize a pre-composed TubeWyrm that operates in a similar manner
        as the SeisBench.models.WaveformModel.annotate() API with the following
        elements

            SBMTubeWyrm
        WindowWyrm -> MethodWyrm(s) -> SBMWyrm
                      - treat_gaps
                      - sync_traces
                      - apply_fill_rule
                      - normalize

        The :class: `~seisbench.models.WaveformModel` `model` is used to inform...
        
            :class: `~ayahos.wyrms.windowwyrm.WindowWyrm`
                :meth: `~ayahos.wyrms.windowwyrm.WindowWyrm().update_from_seisbench(model = model)`
        
            :class: `~ayahos.wyrms.methodwyrm.MethodWyrm`

                MethodWyrm(s) for:
                    "treat_gaps"
                        pkwargs = {'filter_kwargs': {'type': model.filter_args}.update({model.filter_kwargs})}
                    "normalize_traces"
                        pkwargs = {'norm_type': model.norm}
                      
            :class: `~ayahos.wyrms.sbmwyrm.SBMWyrm`
                ayahos.wyrms.sbmwyrm.SBMWyrm(model = model,
                                             weight_names = weight_names,
                                             devicetype = devicetype,
                                             min_batch_size=1,
                                             max_batch_size=model._annotate_args['batch_size'][1])

        This submodule provides access to a minimum number of additional arguments
        that are not carried by a SeisBench WaveformModel object in order to simplify
        specifications. 
            
        :param model: model architecture object, defaults to sbm.EQTransformer()
        :type model: seisbench.models.WaveformModel, optional
        :param weight_names: list of pretrained model weights loadble with model.from_pretrained(), defaults to ['pnw']
        :type weight_names: list of str, optional
            see :meth: `~seisbench.models.WaveformModel.from_pretrained`
        :param devicetype: PyTorch device type, defaults to 'cpu'
        :type devicetype: str, optional
            see :class: `~torch.device`
        :param compiled: should the model(s) contained in SBMWyrm, defaults to False
        :type compiled: bool
            see :class: `~ayahos.wyrms.sbmwyrm.SBMWyrm`
                :meth: `~torch.compile`
        :param fill_rule: missing channel fill rule for <3-component instruments, defaults to 'zeros'
        :type fill_rule: str, optional
            see :meth: `~ayahos.core.windowstream.WindowStream.treat_gaps`
        """
        # Inherit from TubeWyrm
        super().__init__(max_pulse_size=1, wait_sec=0.)

        # Compatability check for model
        if isinstance(model, sbm.WaveformModel):
            if model.name != 'WaveformModel':
                self.model = model

        # Initialize a WindowWyrm object with defaults
        windwyrm = WindowWyrm()
        # Update parameters using the seisbench model's attributes
        windwyrm.update_from_seisbench(model=self.model)

        # Initialize MethodWyrm objects for preprocessing
        if self.model.filter_args is not None:
            filter_kwargs = {'type': self.model.fitler_args}.update(self.model.filter_kwargs)
            
        else:
            filter_kwargs = {}
        # For treating gappy data
        mwyrm_gaps = MethodWyrm(
            pclass=WindowStream,
            pmethod='treat_gaps',
            pkwargs={'filterkw': filter_kwargs})
        
        # For synchronizing temporal sampling and windowing
        mwyrm_sync = MethodWyrm(
            pclass=WindowStream,
            pmethod='sync_to_reference',
            pkwargs={})

        # For filling data out to 3-C from non-3-C data (either missing channels, or 1C instruments)
        mwyrm_fill = MethodWyrm(
            pclass=WindowStream,
            pmethod='apply_fill_rule',
            pkwargs={'rule': fill_rule})

        # Initialize model specific normalization MethodWyrms
        mwyrm_norm = MethodWyrm(
            pclass=WindowStream,
            pmethod='normalize_traces',
            pkwargs={'norm_type': self.model.norm}
        )

        # Initialize MLDetectWyrm object for prediction work
        sbmwyrm = SBMWyrm(model=model,
                            weight_names=weight_names,
                            devicetype=devicetype,
                            min_batch_size=min_batch_size,
                            max_batch_size=model._annotate_args['batch_size'][1]
                            )

        wyrm_dict = {'generate_windows': windwyrm,
                     'degap_detrend_resample': mwyrm_gaps,
                     'sync_sampling': mwyrm_sync,
                     'fill_missing_components': mwyrm_fill,
                     'normalize_traces': mwyrm_norm,
                     'ml_predict': sbmwyrm}

        # Initialize TubeWyrm inheritance
        super().__init__(
            wyrm_dict = wyrm_dict,
            max_pulse_size=max_pulse_size
            )
    

