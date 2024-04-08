"""
:module: util.time
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0

:purpose:
    This module contains methods assisting with different time formats
:attribution:
    This module modifies the `unix_to_true_time` method
    written by Alex Hutko (PNSN) with added annotations by N. Stevens

"""
from obspy import UTCDateTime
from pandas import Timestamp
from pyrocko.gui.marker import Marker
from datetime import datetime


def unix_to_epoch(unixtime, output_format=float):
    """
    Convert Unix timestamps into epoch times, correcting for leap seconds

    Based on the `unix_to_true_time` method by Alex Hutko (PNSN)

    :: INPUT ::
    :param unixtime: [float]
            Unix timestamp
    :param output_format: [method]
            Output as datetime.datetime?
    
    :: OUTPUT ::
    :return epochtime: [float] or [type(output_format(epoch_time))]
                    Epoch timestamp with leap-second corrections
    """
    leap_seconds = {
        2272060800: 10,
        2287785600: 11,
        2303683200: 12,
        2335219200: 13,
        2366755200: 14,
        2398291200: 15,
        2429913600: 16,
        2461449600: 17,
        2492985600: 18,
        2524521600: 19,
        2571782400: 20,
        2603318400: 21,
        2634854400: 22,
        2698012800: 23,
        2776982400: 24,
        2840140800: 25,
        2871676800: 26,
        2918937600: 27,
        2950473600: 28,
        2982009600: 29,
        3029443200: 30,
        3076704000: 31,
        3124137600: 32,
        3345062400: 33,
        3439756800: 34,
        3550089600: 35,
        3644697600: 36,
        3692217600: 37,
    }
    time1900 = unixtime + 2208988800
    seconds_to_sub = 0
    for utime in leap_seconds:
        if time1900 >= utime:
            seconds_to_sub = leap_seconds[utime] - 10
    epochtime = unixtime - seconds_to_sub
    epochtime = output_format(epochtime)
    return epochtime


def unix_to_UTCDateTime(unix):
    return UTCDateTime(unix_to_epoch(unix))

def UTCDateTime_to_Timestamp(utcdatetime):
    """
    Convenience method for translating from
    obspy.core.utcdatetime.UTCDateTime to
    pandas.Timestamp
    """
    return Timestamp(utcdatetime.isoformat())


def Timestamp_to_UTCDateTime(timestamp):   
    """
    Convenience method for translating from
    pandas.Timestamp to
    obspy.core.utcdatetime.UTCDateTime
    """
    return UTCDateTime(timestamp.isoformat())


def format_timestamp(pick_object, unix=False):
    """
    Extract an epoch timestamp from a variety of pick object formats
    :: INPUT ::
    :param pick_object: Currently handles:
                        obspy.core.utcdatetime.UTCDateTime
                        pandas._libs.tslibs.timestamps.Timestamp
                        pyrocko.gui.markers.Marker (and child-classes)
                        datetime.datetime
    :: OUTPUT ::
    :return time: [float] epoch time
    """
    if isinstance(pick_object, UTCDateTime):
        time = pick_object.timestamp
    elif isinstance(pick_object, Timestamp):
        time = pick_object.timestamp()
    elif isinstance(pick_object, Marker):
        time1 = pick_object.get_tmin()
        time2 = pick_object.get_tmax()
        if time1 == time2:
            time = time1
        else:
            time = (time1 + time2) / 2
    elif isinstance(pick_object, datetime):
        time = pick_object
    else:
        NotImplementedError(f"pick_object of type {type(pick_object)} not supported")
    if unix:
        time = unix_to_epoch(time)

    return time