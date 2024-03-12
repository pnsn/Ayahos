from wyrm.core.process import MethodWyrm
from wyrm.data.dictstream import ComponentStream

class GapFillWyrm(TubeWyrm):

    def __init__(
            self,
            filterkw=False, 
            resample_method='resample',
            resample_kw={'sampling_rate': 100.},
            taper_kw={'max_percentage': None, 'max_length': 0.06, 'side': 'both'},
            merge_kw={'method': 1}):

        # Compose ProcessWyrms
        if filterkw:
            filter_wyrm = MethodWyrm(pmethod='filter'
                                     pkwargs=filterkw)
        # Detrending

        # Resampling
        resample_wyrm = MethodWyrm(pmethod=resample_method,
                                   pkwargs=resample_kw)
        
        # 