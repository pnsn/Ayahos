"""
:module: wyrm.util.earthworm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: This module (under development) will host methods for loading information
        from the mapped earthworm config files (*.d and *.desc) to serve as cross-checks against PyEW and 
        Ayohos
"""
import argparse, configparser, logging
import numpy as np

Logger = logging.getLogger('ayahos.util.earthworm.py')

def read_config_file(conffile):
    config = configparser.ConfigParser()
    config.read(conffile)
    return config

def parse_wyrm_config_section(config, section):
    required_fields = ['CLASS']
    if section not in config.keys():
        raise KeyError(f"{section} is not in config's keys")
    elif not all(x in config[section].keys() for x in required_fields):
        raise KeyError(f"not all required keys are present in this section")
    else:
        pass
        cfgsect = config[section]
# Check if class is imported
    if cfgsect['CLASS'] not in dir():
        Logger.critical(f'{cfgsect['CLASS']} not in current scope')
    else:
        pass

    kwargs = {}
    for _k, _v in cfgsect:
        # Pass on CLASS
        if _k.upper() == 'CLASS':
            iclass = _v
            continue
        # Parse float/int arguments
        elif _v.isnumeric():
            _v = float(_v)
            if _v == int(_v):
                _v = int(_v)
        # Parse dictionary like entries
        elif '{' in _v and '}' in _v:
            _v = eval(_v)
            for _k2, _v2 in _v.items():
                if _v2.isnumeric():
                    _v2 = float(_v2)
                    if _v2 == int(_v2):
                        _v2 = int(_v2)
        kwargs.update({_k:_v})
    instruction = {iclass: kwargs}
    return instruction
        


############################################
# Methods for writing results to EarthWorm #
############################################


def _pick_quality_mapping(
    X, grade_max=(0.02, 0.03, 0.04, 0.05, np.inf), dtype=np.int32
):
    """
    Provide a mapping function between a continuous parameter X
    and a discrete set of grade bins defined by their upper bounds

    :: INPUTS ::
    :param X: [float] input parameter
    :param grade_max: [5-tuple] monotonitcally increasing set of
                    values that provide upper bounding values for
                    a set of bins
    :param dtype: [type-assignment method] default nunpy.int32
                    type formatting to assign to output

    :: OUTPUT ::
    :return grade: [dtype value] grade assigned to value

    """
    # Provide sanity check that INF is included as the last upper bound
    if grade_max[-1] < np.inf:
        grade_max.append(np.inf)
    for _i, _g in enumerate(grade_max):
        if X <= _g:
            grade = dtype(_i)
            break
        else:
            grade = np.nan
    return grade


def ml_prob_models_to_PICK2K_msg(
    feature_dataframe,
    pick_metric="et_mean",
    std_metric="et_std",
    qual_metric="",
    m_type=255,
    mod_id=123,
    org_id=1,
    seq_no=0,
):
    """
    Convert ml pick probability models into Earthworm PICK2K formatted
    message strings
    -- FIELDS --
    1.  INT Message Type (1-255)
    2.  INT Module ID that produced this message: codes 1-255 signifying, e.g., pick_ew, PhaseWorm
    3.  INT Originating installation ID (1-255)
    4.  Intentional 1 blank (i.e., ‘ ‘)
    5.  INT Sequence # assigned by picker (0-9999). Key to associate with coda info.
    6.  Intentional 1 blank
    7.  STR Site code (left justified)
    8.  STR Network code (left justified)
    9.  STR Component code (left justified)
    10. STR Polarity of first break
    11. INT Assigned pick quality (0-4)
    12. 2 blanks OR space for phase assignment (i.e., default is ‘  ‘)
    13. INT Year
    14. INT Month
    15. INT Day
    16. INT Hour
    17. INT Minute
    18. INT Second
    19. INT Millisecond
    20. INT Amplitude of 1st peak after arrival (counts?)
    21. INT Amplitude of 2nd peak after arrival
    22. INT Amplitude of 3rd peak after arrival
    23. Newline character

    """
    msg_list = []
    for _i in range(len(feature_dataframe)):
        _f_series = feature_dataframe.iloc[_i, :]
        grade = _pick_quality_mapping(_f_series[qual_metric])
        # Fields 1, 2, 3, 4
        fstring = f"{m_type:3d}{mod_id:3d}{org_id:3d} "
        # Fields 5 - 8
        fstring += f"{seq_no:4d} {_f_series.sta:-5s}{_f_series.net:-2s}"
        fstring += f"{_f_series.cha:-3s} "
        # Fields 10 -
        # fstring += f' {}
        msg_list.append(fstring)

    return msg_list 