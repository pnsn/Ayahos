from wyrm.core.wyrm import Wyrm, EarWyrm, ObsWyrm, WindWyrm


    


class TubeWyrm(TubeWyrm):
    """
    BaseClass for holding a set of wyrm objects and providing a pulse method
    where an initialized PyEW.EWModule (module) and pre-established connection indices
    for input (in_idx) and output (out_idx) 
    """
    def __init__(self, wyrm_list, module=None, in_idx=None, out_idx=None):
        # Run compatability checks on wyrm_list
        if isinstance(wyrm_list, Wyrm):
            self.wyrm_list = [wyrm_list]
        elif isinstance(wyrm_list, list):
            if all(isinstance(_wyrm, Wyrm) for _wyrm in wyrm_list):
                self.wyrm_list = wyrm_list
            else:
                print('TypeError: Not all objects in wyrm_list are type <class Wyrm>!')
                raise TypeError
        # Run compatability checks on module
        if isinstance(module, (type(None), EWModule)):
            self.module = module
        else:
            raise TypeError
        # Run compatability checks on in_idx
        try:
            self.in_idx = int(in_idx)
        except:
        # Run compatability checks on out_idx
        self.out_idx = out_idx

    def __repr__(self):
        rstr = '----- EW Connection -----\n'
        rstr += f'Module: {self.module}\n'
        rstr += f'IN IDX: {self.in_idx}\n'
        rstr += f'OUT IDX: {self.out_idx}\n'
        rstr += '----- Wyrm List -----\n'
        for _wyrm in self.wyrm_list:
            rstr += f'{_wyrm}\n'
        return rstr

    def pulse(self, x=None):
        """
        Send pulse command to sequential wyrms in wyrm_list, passing
        the output of one wyrm.pulse(x) to the input of the next
        wyrm.pulse(x).

        

        :: INPUT ::
        :param x: [None] Input to pass to the first wyrm in self.wyrm_list
                    Default is None
        :: OUTPUT
        """
        if len(self.wyrm_list) == 1:
            y = self.wyrm_list(
                x,
                ewmod=self.module,
                in_idx=self.in_idx,
                out_idx=self.out_idx
                )
        elif len(self.wyrm_list) > 1:
            for _i, _wyrm in enumerate(self.wyrm_list):
                if _i == 0:
                    x = _wyrm.pulse(
                        x,
                        ewmod=self.module,
                        in_idx=self.in_idx,
                        out_idx=None
                        )
                elif _i == len(self.wyrm_list) - 1:
                    x = _wyrm.pulse(
                        x,
                        ewmod=self.module,
                        in_idx=None,
                        out_idx=self.out_idx
                        )
                else:
                    x = _wyrm.pulse(x, ewmod=None, in_idx=None, out_idx=None)
            elif _i == len
            x = _wyrm.pulse(x, ewmod=ewmod, connidx=conn_list[_i])
        y = x
        return y