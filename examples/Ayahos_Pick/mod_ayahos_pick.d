#
# mod_ayahos_pick.data
#
# Parameter File for MOD_AYAHOS_PICK
# Parsed using python's configparser module
# note: indentation is purely stylistic for readability

# Earthworm Specific Parameters
[Earthworm]
    MOD_ID = 200
    INST_ID = 2
    HB = 30
    WAVE_RING = 1000
    PICK_RING = 1005

# Machine Learning Model
[model]
    obj='sbm.EQTransformer()'
    weight_names=['pnw','instance']
    blinding=(500,500)
    overlap=3000

# Ayahos Module Initialzation Parameters
[Ayahos]
    default_ring_id = int(${Earthworm:WAVE_RING})
    module_id = int(${Earthworm:MOD_ID})
    installation_id = int(${Earthworm:INST_ID})
    heartbeat_period = float(${Earthworm: HB})
    extra_connections = {'WAVE_RING': ${Earthworm:IN_RING}, 'PICK_RING': ${Earthworm:OUT_RING}}
    module_debug = 'False'

# Wyrm submodule initialization Parameters
[WaveInWyrm]
    conn_name =WAVE_RING
    max_buffer_length = 300.

# Seisbench Model TubeWyrm init parameters
[SBMTubeWyrm]
    weight_names = ['pnw','instance']

# Pickout init parameters
[PickOutWyrm]
    conn_name = PICK_RING
    max_buffer_length = 120.
    buffer_append_method=1
    prestack_blinding=${model:blinding}
    leading_mute=${model:overlap}
