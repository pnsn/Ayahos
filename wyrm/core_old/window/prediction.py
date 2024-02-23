"""
:module: wyrm.core.window.prediction
:author: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose: 
    This module hosts the class definition for a PredictionWindow


TODO: 
    1) Remove model as input and make **options explicit
    2) Add method to update from SeisBench
    3) Ensure that the outputs from split_for_ml() can be used as
        **kwarg inputs to populate a new PredictionWindow
"""


import numpy as np
import seisbench.models as sbm
import torch
from obspy import UTCDateTime, Stream, Trace
import wyrm.util.compatability as wcc
from copy import deepcopy


class PredictionWindow(object):
    """
    This object houses a data array and metadata attributes for documenting a pre-processed
    data window prior to ML prediction and predicted values by an ML operation. It provides
    options for converting contained (meta)data into various formats.
    """
    def __init__(self,
        data=None,
        id='..--.',
        t0=0.,
        samprate=1.,
        blinding=0,
        labels=[],
        model_name=None,
        weight_name=None
        ):
        """
        Initialize a PredictionWindow (pwind) object

        :: INPUTS ::
        :param data: [numpy.ndarray] or None
                    Data array to house in this PredictionWindow. Must be a
                    2-dimensional array with one axis that has the same number
                    of entries (rows or columns) as there are iterable items
                    in `labels`. This axis will be assigned as the 0-axis.
        :param id: [str] instrument ID code for this window. Must conform to the
                        N.S.L.bi notation (minimally "..."), where:
                            N = network code (2 characters max)
                            S = station code (5 characters max)
                            L = location code (2 characters)
                            bi = band and instrument characters for SEED naming conventions
        :param t0: [float] timestamp for first sample (i.e., seconds since 1970-01-01T00:00:00)
        :param samprate: [float] sampling rate in samples per second
        :param blinding: [int] number of samples to blind on the left and right ends of this window
                            when stacking sequential prediction windows
        :param labels: [list-like] of [str] labels to associate with row vectors in data.
                            Must match the 0-axis of self.data.
        :param model_name: [str] name of the ML model this window is associated with this pwind
        :param weight_name: [str] name of the pretrained model weights associated with this pwind
        :
        """
        # data compat. check
        if not isinstance(data, (type(None), np.ndarray)):
            raise TypeError('data must be type numpy.ndarray or NoneType')
        elif data is None:
            self.data = data
        elif data.ndim != 2:
            raise IndexError(f'Expected a 2-dimensional array. Got {data.ndim}-d')
        else:
            self.data = data
        # id compat. check
        if not isinstance(id, (str, type(None))):
            raise TypeError('id must be type str or NoneType')
        elif id is None:
            self.id = None
        elif len(id.split('.')) == 4:
            self.id = id
        else:
            raise SyntaxError('str-type id must consit of 4 "." delimited string elements to match N.S.L.C notation')
        # t0 compat check
        if not isinstance(t0, (int, float)):
            raise TypeError('t0 must be type int, float, or obspy.core.utcdatetime.UTCDateTime')
        elif isinstance(t0, UTCDateTime):
            self.t0 = t0.timestamp
        else:
            self.t0 = float(t0)
        # samprate compat checks
        self.samprate = wcc.bounded_floatlike(
            samprate,
            name='samprate',
            minimum=0,
            maximum=None,
            inclusive=False
        )
        # blinding compat checks
        self.blinding = wcc.bounded_intlike(
            blinding,
            name='blinding',
            minimum=0,
            maximum=None,
            inclusive=True
        )
        # labels compat check
        if not isinstance(labels, (list, tuple, str)):
            raise TypeError('labels must be type list, tuple, or str')
        elif isinstance(labels, (tuple, str)):
            labels = [_l for _l in labels]
        # labels/data crosschecks
        if self.data is not None:
            if len(labels) == self.data.shape[0]:
                self.labels = labels
            elif len(labels == self.data.shape[1]):
                self.data = self.data.T
                self.labels = labels
            else:
                raise IndexError(f'Number of labels ({len(labels)}) not in self.data.shape ({self.data.shape})')
        else:
            self.labels = labels
        # model_name compat check
        if not isinstance(model_name, (str, type(None))):
            raise TypeError('model_name must be type str or NoneType')
        else:
            self.model_name = model_name
        # weight_name compat check
        if not isinstance(weight_name, (str, type(None))):
            raise TypeError('weight_name must be type str or NoneType')
        else:
            self.weight_name = weight_name

    def get_metadata(self):
        """
        Return a dictionary containing metadata attributes from this PredictionWindow

        :: OUTPUT ::
        :return meta: [dict] metadata dictionary containing the 
            following attributes as key:value pairs (note: any of these may also have a None value)
            'id': [str] - station/instrument ID
            't0': [float] - starttime of windowed data (epoch seconds / timestamp)
            'samprate': [float] - sampling rate of windowed data (samples per second)
            'model_name': [str] or None - name of ML model this window corresponds to
            'weight_name': [str] or None - name of pretrained ML model weights assocaited with this window
            'labels': [list] of [str] - string names of model/data labels
        """
        meta = {'id': self.id,
                't0': self.t0,
                'samprate': self.samprate,
                'model_name': self.model_name,
                'weight_name': self.weight_name,
                'labels': self.labels,
                'blinding': self.blinding}
        return meta
    
    def to_stream(self, apply_blinding=False):
        """
        Return an obspy.core.stream.Stream representation of the data and labels in this PredictionWindow
        The first letter of each label is appended to the end of the channel name for each trace in the
        output Stream object


        :: INPUT ::
        :param apply_blinding: [bool] should data vectors be converted into masked arrays with
                                     masked values in blinded locations?

        :: OUTPUT ::
        :return st: [obspy.core.stream.Stream]
        """
        st = Stream()
        header = dict(zip(['network','station','location','channel'], self.id.split('.')))
        header.update({'starttime': UTCDateTime(self.t0), 'sampling_rate': self.samprate})
        for _i, _l in enumerate(self.labels):
            _data = self.data[_i, :].copy()
            if apply_blinding:
                _data = np.ma.masked(data=_data, mask=np.full(shape=_data.shape, fill_value=False))
                _data.mask[:self.blinding] = True
                _data.mask[-self.blinding:] = True
            tr = Trace(data=_data, header=header)
            tr.stats.channel += _l[0].upper()
            st += tr
        return st

    def split_for_ml(self):
        """
        Convenience method for splitting the data and metadata in this PredictionWindow into
        a pytorch Tensor and a metadata dictionary 

        :: OUTPUTS ::
        :return tensor: [torch.Tensor] Tensor formatted version of self.data
        :return meta: [dict] metadata dictionary - see self.get_metadata()
        """
        meta = self.get_metadata()
        tensor = torch.Tensor(self.data.copy())
        return tensor, meta

    def copy(self):
        """
        Return a deepcopy of this PredictionWindow
        """
        return deepcopy(self)

    def __repr__(self):
        rstr =  'Prediction Window'
        rstr += f'{self.id} | t0: {UTCDateTime(self.t0)} | S/R: {self.samprate: .3f} sps | Dims: {self.shape}\n'
        rstr += f'Model: {self.model_name} | Weight: {self.weight_name} | Labels: {self.labels} | Blind: {self.blinding} \n'
        if self._blank:
            rstr += f'np.zeros(shape={self.shape})'
        else:
            rstr += f'{self.data.__repr__()}'
        return rstr
    
    def __str__(self):
        rstr = 'wyrm.core.window.prediction.PredictionWindow('
        rstr += f'model_name={self.model_name}, weight_name={self.weight_name}, '
        if self._blank:
            rstr += f'data=None, '
        else:
            rstr += f'data={self.data}, '
        rstr += f'id={self.id}, samprate={self.samprate}, t0={self.t0}, labels={self.labels})'
        return rstr

    def __eq__(self, other, include_flags=False):
        """
        Rich representation of self == other. Includes an option to include
        status flags in the comparison.

        :: INPUT ::
        :param other: [object]
        :param include_flags: [bool] - include self._blank == other._blank in comparison?

        :: OUTPUT ::
        :return status: [bool]
        """
        bool_list = []
        if not isinstance(other, PredictionWindow):
            status = False
        else:
            bool_list = [(self.data == other.data).all()]
            for _attr in ['t0','model_name','weight_name','samprate','labels','shape']:
                bool_list.append(eval(f'self.{_attr} == other.{_attr}'))
            if include_flags:
                for _attr in ['_blank']:
                    bool_list.append(eval(f'self.{_attr} == other.{_attr}'))

            if all(bool_list):
                status = True
            else:
                status = False 
        return status

    def update_from_seisbench(self, model, weight_name=None, is_prediction=True):
        """
        Update attributes with a seisbench.model.WaveformModel child-class object
        if they pass compatability checks with the current self.data array in
        this PredictionWindow

        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel] child-class object of the 
                        WaveformModel baseclass (e.g., EQTransformer)
        :param weight_name: [str] or [None] weight name to update self.weight_name
                        with (None input indicates no change)
        :param is_prediction: [bool] use the labels for predictions (True) or inputs (False)
                            True - data axis 0 scaled by len(model.labels)
                            False - data axis 0 scaled by len(model.component_order)
        :: OUTPUT ::
        :return self: [wyrm.core.window.prediction.PredictionWindow] enable cascading
        """
        # model compat check
        if not isinstance(model, sbm.WaveformModel):
            raise TypeError('model must be type seisbench.models.WaveformModel')
        # weight_name compat check
        if not isinstance(weight_name, (str, type(None))):
            raise TypeError('weight_name must be type str or NoneType')
        # dim0 compat check
        if not isinstance(is_prediction, bool):
            raise TypeError('is_prediction must be type bool')
        
        if not is_prediction:
            dim0 = model.component_order
            d0name = 'component_order'
        else:
            dim0 = model.labels
            d0name = 'labels'
        # Fetch dim1
        dim1 = model.in_samples
        # Check that the input model has viable data/prediction array dimensions
        if dim0 is None or dim1 is None:
            raise ValueError(f'model.{d0name} and/or model.in_samples are None - canceling update - suspect model is a base class WaveformModel, rather than a child class thereof')
        else:
            # If data is None, populate
            if self.data is None:
                self.data = np.zeros(shape=(len(dim0), dim1), dtype=np.float32)
            else:
                if self.data.shape != (len(dim0), dim1):
                    raise IndexError(f'proposed data array dimensions from model ({len(dim0)},{dim1}) do not match the current data array ({self.data.shape})')
            # Update weight_name if type str
            if isinstance(weight_name, str):
                self.weight_name = weight_name
            # Update labels
            self.labels = [_l for _l in dim0]
            # Update blinding
            bt = model._annotate_args['blinding'][1]
            if all(np.mean(bt) == _b for _b in bt):
                self.blinding = int(np.mean(bt))
            else:
                raise ValueError('model._annotate_args["blinding"] with mixed blinding values not supported.')
            # Update samprate
            self.samprate = model.sampling_rate
        return self