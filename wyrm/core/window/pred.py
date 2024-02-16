class PredWindow(object):
    """
    Support class to house predicted value arrays and windowing/model metadata
    into a single object used to append to PredBuff
    """
    def __init__(self, data=np.array([]), header={}):
        # Run compatability check on data
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be type numpy.ndarray')
        else:
            self.data = data
        # Map some attributes from self.data to self for convenience
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        
        # Compatability check
        if not isinstance(header, dict):
            raise TypeError
        # Parse id from header dict
        if 'id' in header.keys():
            if isinstance(header['id'], str):
                self.id = header['id']
            else:
                raise TypeError
        else:
            self.id = None
        # Parse 't0' from header dict
        if 't0' in header.keys():
            if isinstance(header['t0'], (float, int)):
                self.t0 = float(header['t0'])
            elif isinstance(header['t0'], UTCDateTime):
                self.t0 = header['t0'].timestamp
            else:
                raise TypeError
        else:
            self.t0 = None

        # Parse 'samprate'
        if 'samprate' in header.keys():
            if isinstance(header['samprate'], (float, int)):
                self.samprate = header['samprate']
            else:
                raise TypeError
        else:
            self.samprate = None

        # Parse 'model_name'
        if 'model_name' in header.keys():
            if isinstance(header['model_name'], str):
                self.model_name = header['model_name']
            else:
                raise TypeError
        else:
            self.model_name = None

        # Parse 'weight_name'
        if 'weight_name' in header.keys():
            if isinstance(header['weight_name'], (str, type(None))):
                self.weight_name = header['weight_name']
            else:
                raise TypeError
        else:
            self.weight_name = None
            
        # Parse 'label_names'
        if 'label_names' in header.keys():
            if isinstance(header['label_names'], str):
                if self.shape[0] == 1:
                    self.labels = {header['label_names']: 0}
            elif not isinstance(header['label_names'], (list, tuple)):
                raise TypeError
            elif not all(isinstance(_l, str) for _l in header['label_names']):
                raise TypeError
            elif len(header['label_names']) != self.shape[0]:
                raise IndexError
            else:
                self.labels = {_l: _i for _i, _l in enumerate(header['label_names'])}

    def update_labels(self, update_dict):
        
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
        Return a dictionary containing metadata attributes of this PredArray

        :: OUTPUT ::
        :return meta: [dict] metadata dictionary containing the 
            following attributes as key:value pairs (note: any of these may also have a None value)
            'id': [str] - station/instrument ID
            't0': [float] - starttime of windowed data (epoch seconds / timestamp)
            'samprate': [float] - sampling rate of windowed data (samples per second)
            'model_name': []
        """
        meta = {'id': self.id,
                't0': self.t0,
                'samprate': self.samprate,
                'model_name': self.model_name,
                'weight_name': self.weight_name,
                'labels': self.labels.copy()}
        return meta