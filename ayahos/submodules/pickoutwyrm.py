from ayahos.wyrms import BufferWyrm, PickWyrm, RingWyrm, TubeWyrm


class PickOutWyrm(TubeWyrm):
    """This convenience class stitches together the following Wyrm modules:

    BufferWyrm -------------> PickWyrm ------> RingWyrm
    - buffer ML timeseries   pick phases       Submit TYPE_PICK2K messages to EW

    """
    def __init__(
            self,
            module,
            module_id,
            installation_id=255,
            conn_id = 1,
            pre_buffer_blinding=(500,500),
            max_buffer_length=120,
            buffer_append_method=3,
            buffer_kwargs={},
            pick_method='max',
            pick_kwargs={},
            ring_kwargs={},
            tube_kwargs={}
            ):
        # Initialize bufferwyrm
        if "max_length" in buffer_kwargs:
            buffer_kwargs.pop("max_length")
        if "blinding" in buffer_kwargs:
            buffer_kwargs.pop("blinding")
        if "method" in buffer_kwargs:
            buffer_kwargs.pop("method")

        buffwyrm = BufferWyrm(
            max_length=max_buffer_length,
            blinding=pre_buffer_blinding,
            method=buffer_append_method,
            **buffer_kwargs
        )

        if "pick_method" in pick_kwargs.keys():
            pick_kwargs.pop('pick_method')
        if "installation_id" in pick_kwargs.keys():
            pick_kwargs.pop('installation_id')

        pickwyrm = PickWyrm(
            module_id,
            installation_id = installation_id,
            pick_method=pick_method,
            **pick_kwargs
        )

        if "conn_id" in ring_kwargs.keys():
            ring_kwargs.pop('conn_id')
        if 'pulse_method' in ring_kwargs.keys():
            ring_kwargs.pop('pulse_method')
        if 'msg_type' in ring_kwargs.keys():
            ring_kwargs.pop('msg_type')

        ringwyrm = RingWyrm(
            module,
            conn_id = conn_id,
            pulse_method='put_msg',
            msg_type=10,
            **ring_kwargs
        )

        super().__init__(wyrm_dict={'predbuff': buffwyrm,
                                    'pick': pickwyrm,
                                    'put_msg': ringwyrm},
                         **tube_kwargs)
