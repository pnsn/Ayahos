"""
:module: camper.module.trigger
:author: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This conatins the class definition for a module that facilitates triggering on
    :class:`~camper.data.trace.Trace` objects containing characteristic response function
    (ML phase pick probability) time series and conversion of triggers into :class:`~camper.data.pick.Pick`
    objects.

Classes
-------
:class:`~camper.module.unit.trigger.TriggerMod`
"""