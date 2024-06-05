from ayahos import BufferWyrm, RingWyrm, TubeWyrm

class WaveInWyrm(TubeWyrm):
    """
    This class provides a convenience combination of a RingWyrm set to get_wave for a specified Earthworm memory ring
    that feeds to a BufferWyrm, housed in a TubeWyrm-like object. The default parameters fed to the Ring and Buffer init
    arguments set WaveInWyrm to pass a read message from the EW wave ring directly to the BufferWyrm, rather than
    aggregating numerous wave messages.


    :param TubeWyrm: _description_
    :type TubeWyrm: _type_
    """
    def __init__(
            self,
            module,
            conn_id=0,
            max_buffer_length=300,
            wait_sec=0.0,
            max_pulse_size=10000,
            ringwyrm_init={'max_pulse_size': 1},
            bufferwyrm_init={'max_pulse_size': 1}):
        
        # Initialize RingWyrm object
        if isinstance(ringwyrm_init, type(None)):
            ringwyrm_init = {}
        elif not isinstance(ringwyrm_init, dict):
            raise TypeError
        if 'module' in ringwyrm_init.key():
            raise KeyError
        else:
            ringwyrm_init.update({"module": module})

        if 'conn_id' in ringwyrm_init.keys():
            raise KeyError
        else:
            ringwyrm_init.update({"conn_id":conn_id})
        ringwyrm = RingWyrm(**ringwyrm_init)

        # Initialize BufferWyrm object
        if isinstance(bufferwyrm_init, type(None)):
            bufferwyrm_init = {}
        elif not isinstance(bufferwyrm_init, dict):
            raise TypeError
        if 'max_length' in bufferwyrm_init.keys():
            raise KeyError
        else:
            bufferwyrm_init.update({'max_length': max_buffer_length})
        
        bufferwyrm = BufferWyrm(**bufferwyrm_init)

        # Form WyrmDict
        wyrm_dict = {'get_waves': ringwyrm,
                     'buffer_waves': bufferwyrm}
        # Initialize TubeWyrm inheritance
        super().__init__(
            wyrm_dict=wyrm_dict,
            wait_sec=wait_sec,
            max_pulse_size=max_pulse_size)