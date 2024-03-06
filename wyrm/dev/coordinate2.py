from collections import deque
from wyrm.core._base import Wyrm
from wyrm.dev.trace import MLTrace, MLTraceBuffer
from wyrm.dev.dictstream import DictStream
import wyrm.util.compatability as wuc


class BufferWyrm(Wyrm):

    def __init__(self, max_length=300., add_style='stack', restrict_add_past=True, max_pulse_size=10000, debug=False, **add_kwargs):

        self.max_length = wuc.bounded_floatlike(
            max_length,
            name='max_length',
            minimum = 0,
            maximum=None,
            inclusive=False
            )

        super().__init__(max_pulse_size=max_pulse_size, debug=debug)

        if add_style not in ['stack','merge']:
            raise ValueError(f'add_style "{add_style}" not supported. "stack" and "merge" only')
        else:
            self.add_style=add_style
        

        _stack_kwargs = {'blinding_samples': int,'method': str,'fill_value': (type(None), int, float),'sanity_checks': bool}
        _merge_kwargs = {'interpolation_samples': int,'method': str,'fill_value': (type(None), int, float),'sanity_checks': bool}
        _kwarg_checks = {'stack': _stack_kwargs, 'merge': _merge_kwargs}

        for _k, _v in add_kwargs.items():
            if _k not in _kwarg_checks[self.add_style].keys():
                raise KeyError(f'kwarg {_k} is not supported for add_style "{self.add_style}"')
            elif not isinstance(_v, _kwarg_checks[self.add_style][_k]):
                raise TypeError(f'kwarg {_k} type "{type(_v)}" not supported for add_style "{self.add_style}')
        self.add_kwargs = add_kwargs
            
        if not isinstance(restrict_add_past, bool):
            raise TypeError
        else:
            self.restrict_add_past = restrict_add_past

        self.buffer = DictStream()

    def __repr__(self, extended=False):
        rstr = f'Add Style: {self.add_style}'

    def pulse(self, x, **options):
        if not isinstance(x, deque):
            raise TypeError
        
        qlen = len(x)
        for _i in range(self.max_pulse_size):
            if qlen == 0:
                break
            elif _i + 1 > qlen:
                break
            else:
                _x = x.popleft()
            
            if not isinstance(_x, MLTrace):
                x.append(_x)
            else:
                _id = _x.id
                if _id not in self.buffer.labels():
                    new_buffer_item = MLTraceBuffer(max_length=self.max_length,
                                               add_style=self.add_style,
                                               restrict_add_past=self.restrict_add_past)
                    new_buffer_item.__add__(_x, **self.add_kwargs)
                    self.buffer.append(new_buffer_item, **self.add_kwargs)
                else:
                    self.buffer.append(_x,
                                       restrict_add_past=self.restrict_add_past,
                                        **self.add_kwargs)

        