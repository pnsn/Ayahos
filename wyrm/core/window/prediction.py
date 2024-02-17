import numpy as np
import seisbench.models as sbm
from torch import Tensor
from obspy import UTCDateTime, Stream, Trace
import wyrm.util.compatability as wcc
from copy import deepcopy

class PredictionWindow(object):
    """
    This object houses a data array and metadata attributes for documenting a pre-processed
    data window prior to ML prediction and predicted values by an ML operation. It provides
    options for converting contained (meta)data into various formats.
    """
    def __init__(self, model=None, data=None, id='..--.', samprate=1., t0=0., blinding=(0,0), **options):
        """
        Initialize a PredictionWindow (pwind) object
        
        :: INPUTS ::
        :param model: [seisbench.models.WaveformModel] an initialized
                        sbm.WaveformModel child-class object (e.g., EQTransformer)
                        that has the following attributes populated:
                            @ model.name
                            @ model.in_samples
                            @ model.labels
                    None - these fields are sought out in the 'header' input
                          as keys 'name', 'in_samples', and 'labels'
                    NOTE: if any of these fields show up in `header` the model values
                    will supercede those in header
        
        :param data: [numpy.ndarray] appropriately scaled array with dimensions
                        [1, len(model.labels), len(model.in_samples)]
                     None - populate an zeros array, scaled as above with axis 1
                            length determined as len(model.labels)
        :param id: [str] instrument ID code for this window. Must conform to the
                        N.S.L.bi notation (minimally "..."), where:
                            N = network code (2 characters max)
                            S = station code (5 characters max)
                            L = location code (2 characters)
                            bi = band and instrument characters for SEED naming conventions
        :param t0: [float] timestamp for first sample (i.e., seconds since 1970-01-01T00:00:00)
        :param samprate: [float] sampling rate in samples per second
        :param blinding: [2-tuple] of non-negative [int] - samples to blind on the left (blinding[0])
                            and right (blinding[1]) ends of this window

        OPTIONAL KWARGS (via **options)
        :options model_name: [str] name of ML model
        :options labels: [list] or [str] list of component/label names or an iterable string
                            of single character label names
        :options 
        :options weight_name: [str] name of pretrained model weight set or [None]

        """
        # model compatability checks
        if not isinstance(model, (sbm.WaveformModel, type(None))):
            raise TypeError('model must be type seisbench.models.WaveformModel or NoneType')
        
       
        # data compatability checks
        if not isinstance(data, (np.ndarray, type(None))):
            raise TypeError('data must be type numpy.ndarray or NoneType')
        elif isinstance(data, np.ndarray):
            self._blank = False
            if data.ndim not in [2,3]:
                raise IndexError(f'Expected a 2 or 3-dimensional array')
        else:
            self._blank = True

        # id compat. checks
        if not isinstance('id', (str, type(None))):
            raise TypeError
        elif isinstance(id, str):
            if len(id.split('.')) == 4:
                self.id = id
            else:
                raise SyntaxError('str-type id must consist of 4 "." delimited string segments (including empty string). E,g., "..."')
        else:
            self.id = id

        # t0 compat. checks
        if isinstance(t0, (int, float)):
            self.t0 = float(t0)
        elif isinstance(t0, UTCDateTime):
            self.t0 = t0.timestamp
        elif t0 is None:
            self.t0 = 0.
        else:
            raise TypeError('t0 must be type int, float, obspy.core.utcdatetime.UTCDateTime, or NoneType')
        
        # samprate compat. checks
        self.samprate = wcc.bounded_floatlike(
            samprate,
            name='samprate',
            minimum=0,
            maximum=None,
            inclusive=False)
        
        if not isinstance(blinding, (tuple, list)):
            raise TypeError('blinding must be type tuple or list')
        elif len(blinding) != 2:
            raise ValueError('blinding must be a 2 element tuple or list')
        elif not all(isinstance(_b, int) for _b in blinding):
            raise TypeError('Elements in blinding must be type int')
        elif any(_b < 0 for _b in blinding):
            raise ValueError('Elements in blinding must be non-negative')
        else:
            self.blinding = blinding


        # weight_name compatability checks
        if 'weight_name' in options.keys():
            if not isinstance(options['weight_name'], (str, type(None))):
                raise TypeError('weight_name must be type str or NoneType')
            else:
                self.weight_name = options['weight_name']
        else:
            self.weight_name = None

        # Run sanity checks if model is None
        if model is None:
            if data is None:
                if not all(_k in options.keys() for _k in ['labels','in_samples']):
                    raise KeyError('For model=None and data=None, keys "labels" and "in_samples" are required in header')
            else:
                if 'labels' in options.keys():
                    if len(options['labels']) not in data.shape[:2]:
                        raise ValueError('The length of input kwarg "labels" does not match the length of "data" axes 0 or 1')
                else:
                    raise ValueError('insufficient label information to initialize a PredictionWindow - revise `model` input or add an optional kwarg of `labels`')
                
        # Run model_name ingestion
        if 'model_name' in options.keys():
            if isinstance(options['model_name'], str):
                self.model_name = options['model_name']
            else:
                raise TypeError('option "model_name" must be type str')
        elif model is not None:
            self.model_name = model.name
        else:
            self.model_name = None

        # Run in_samples ingestion
        if model is not None:
            self.in_samples = model.in_samples
        elif 'in_samples' in options.keys():
            self.in_samples = wcc.bounded_intlike(
                options['in_samples'],
                name="in_samples",
                minimum=0,
                maximum=None,
                inclusive=False)
        elif isinstance(data, np.ndarray):
            self.in_samples = data.shape[-1]
        
        # Run labels ingestion
        if model is not None:
            self.labels = [_k for _k in model.labels]
        elif 'labels' in options.keys():
            if isinstance(options['labels'], (list, str)):
                self.labels = [_k for _k in options['labels']]
            else:
                raise TypeError('optional input `labels` must be type str or list')
        
        self._nl=len(self.labels)
        self.shape = (1, self._nl, self.in_samples)

        # Run checks on Data
        if data is None:
            self.data = np.zeros(shape=self.shape, dtype=np.float32)
        else:
            # Provide some support for different dimensionalities    
            if data.ndim == 2:
                if data.shape[0] == self.in_samples and data.shape[1] == self._nl:
                    self.data = data.T[np.newaxis, :, :]
                elif data.shape[1] == self.in_samples and data.shape[0] == self._nl:
                    self.data = data[np.newaxis, :, :]
                else:
                    raise IndexError(f'Input array `data` has an incompatable shape {data.shape} relative to the metadata ({self._nl}, {self.in_samples})')
            # Require some user responsibility for transforming data in R3
            elif data.ndim == 3:
                if data.shape == self.shape:
                    self.data = data
                else:
                    raise IndexError(f'Input array `data` shape {data.shape} does not match expected shape from metadata {self.shape} - consider numpy.moveaxis')
            
    def update_labels(self, update_dict):
        """
        Wrapper for the dict.update() builtin method to facilitate
        changing self.labels keys in-place.

        :: INPUT ::
        :param update_dict: [dict] with {"oldlabel":"newlabel"} formatting

        :: OUTPUT ::
        :return self: [PredictionWindow] enable cascading
        """
        new_keys = []
        for _k in self.labels.keys():
            if _k in update_dict.keys():
                new_keys.append(update_dict[_k])
            else:
                new_keys.append(_k)
        new_labels = dict(zip(new_keys, self.labels.values()))
        self.labels = new_labels

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
        :: OUTPUT ::
        :return st: [obspy.core.stream.Stream]
        """
        st = Stream()
        header = dict(zip(['network','station','location','channel'], self.id.split('.')))
        header.update({'starttime': UTCDateTime(self.t0), 'sampling_rate': self.samprate})
        for _i, _l in enumerate(self.labels):
            data = np.squeeze(self.data[:, _i, :].copy())
            tr = Trace(data=data, header=header)
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
        tensor = Tensor(self.data.copy())
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