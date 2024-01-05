from obspy import Trace, Stream
from wyrm.structures.rtbufftrace import RtBuffTrace


class WindowMsg(Stream):
    """ """

    def __init__(
        self,
        V0=None,
        H1=None,
        H2=None,
        ch_fill_rule="zeros",
        model_code="EQT",
        target_sr=100.0,
    ):
        # Initialize parent class attributes (stream)
        super().__init__(self)

        # V0 compatability checks
        if isinstance(V0, RtBuffTrace):
            self.V0 = V0.as_trace()
        elif isinstance(V0, Trace):
            self.V0 = V0.copy()
        elif V0 is None:
            self.V0 = V0
        else:
            raise TypeError("V0 must be type Trace or None")
        if self.V0 is not None:
            self.traces.append(V0)

        # H1 compatability checks
        if isinstance(H1, RtBuffTrace):
            self.H1 = H1.as_trace()
        elif isinstance(H1, Trace):
            self.H1 = H1.copy()
        elif H1 is None:
            self.H1 = H1
        else:
            raise TypeError("H1 must be type Trace or None")
        if self.H1 is not None:
            self.traces.append(H1)

        # H2 compatability checks
        if isinstance(H2, RtBuffTrace):
            self.H2 = H2.as_trace()
        elif isinstance(H2, Trace):
            self.H2 = H2.copy()
        elif H2 is None:
            self.H2 = H2
        else:
            raise TypeError("H2 must be type Trace or None")
        if self.H2 is not None:
            self.traces.append(H2)

        # ch_fill_rule compatability checks
        if not isinstance(ch_fill_rule, str):
            raise TypeError("ch_fill_rule must be type str")
        elif ch_fill_rule in ["zeros", "cloneZ", "cloneHZ"]:
            self.ch_fill_rule = ch_fill_rule
        else:
            raise ValueError(
                f'ch_fill_rule {ch_fill_rule} not supported. Only "zeros", "cloneZ", or "cloneHZ"'
            )

        # model_code compatability checks
        if not isinstance(model_code, str):
            raise TypeError("model_code must be type str")
        elif model_code in ["EQT", "PN"]:
            self.model_code = model_code
        else:
            raise ValueError(f"model code {model_code} not supported")

        # target_sr compatability checks
        if isinstance(target_sr, (int, float)):
            if target_sr > 0:
                self.target_sr = target_sr
            else:
                raise ValueError("target_sr must be positive")
        else:
            raise TypeError("target_sr must be int or float")

        # Mutual compatability checks
        for _i, _c in enumerate([self.V0, self.H1, self.H2]):
            for _j, _k in enumerate([self.V0, self.H1, self.H2]):
                if _i > _j:
                    if _c is not None and _k is not None:
                        try:
                            self.check_inst_meta_compat(_c, _k)
                        except ValueError:
                            raise ValueError(f"{_c} mismatches {_k}")

        # Handle non 3-C WindowMsg inputs based on ch_fill_rule
        # setting attribute aliases to duplicate data
        # NOTE: contents of self.traces is the definitive list of unique data
        if self.ch_fill_rule == "zeros":
            if self.H1 is None or self.H2 is None:
                self.CH = self.V0.copy()
                self.CH.data *= 0
                self.CH.stats.channel = self.CH.stats.channel[:-1] + "0"
                self.H1 = self.CH
                self.H2 = self.CH
                self.traces.append(self.CH)

        elif self.ch_fill_rule == "cloneZ":
            if self.H1 is None or self.H2 is None:
                self.CH = self.V0.copy()
                self.H1 = self.CH
                self.H2 = self.CH

        elif self.ch_fill_rule == "cloneHZ":
            if self.H1 is None and self.H2 is None:
                self.CH = self.V0.copy()
                self.H1 = self.CH
                self.H2 = self.CH
            elif self.H1 is None:
                self.H1 = self.H2
            elif self.H2 is None:
                self.H2 = self.H1
        # Kick error if ch_fill_rule got changed incorrectly later
        else:
            raise ValueError(f"ch_fill_rule {ch_fill_rule} invalid")

    def __str__(self):
        rstr = f"{len(self)} trace(s) in WindowMsg | "
        rstr += f'target model: {self.model_code} | '
        rstr += f"channel fill rule: {self.ch_fill_rule} | "
        rstr += f'target S/R: {self.target_sr} Hz\n'
        # Vertical component trace display
        rstr += f"Vert:  {self.V0.__str__()} \n"
        # Horizontal component 1 trace display
        rstr += f"Hztl 1:{self.H1.__str__()}"
        # And ch_fill_rule annotations
        if self.H1.stats.channel[-1] == "0":
            rstr += " (blank)\n"
        elif self.H1.stats.channel == self.H2.stats.channel:
            rstr += " (twinned)\n"
        else:
            rstr += "\n"
        # Horizontal component 2 trace display
        rstr += f"Hztl 2:{self.H2.__str__()}"
        # And ch_fill_rule annotations
        if self.H2.stats.channel[-1] == "0":
            rstr += " (blank)\n"
        elif self.H1.stats.channel == self.H2.stats.channel:
            rstr += " (twinned)\n"
        else:
            rstr += "\n"
        return rstr
    
    def __repr__(self):
        rstr = self.__str__()
        return rstr

    def check_inst_meta_compat(self, c1, c2, tolsamp=3):
        if not isinstance(c1, Trace):
            raise TypeError("c1 is not type Trace")
        if not isinstance(c2, Trace):
            raise TypeError("c2 is not type Trace")
        # Get stats
        s1 = c1.stats
        s2 = c2.stats
        # Check stats that must be identical
        for _k in ["station", "network", "location", "sampling_rate", "calib"]:
            if s1[_k] != s2[_k]:
                raise ValueError(f"c1({s1[_k]}) != c2({s2[_k]}) [{_k} test]")
            else:
                pass
        # Check instrument/bands codes match
        if s1["channel"][:-1] != s2["channel"][:-1]:
            raise ValueError(
                f'c1({s1["channel"]}) != c2({s2["channel"]}) [inst/band test]'
            )
        else:
            pass
        # Check that starttime and endtime are within tolerance
        for _k in ["starttime", "endtime"]:
            if abs(s1[_k] - s2[_k]) / s1.sampling_rate > tolsamp:
                raise ValueError(f"difference in {_k}'s outside tolerance")
            else:
                pass

        return True
