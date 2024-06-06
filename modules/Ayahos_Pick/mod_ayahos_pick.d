#
# mod_ayahos_pick.data
#
# Parameter File for MOD_AYAHOS_PICK
# Parsed using python's configparser module
# note: indentation is purely stylistic for readability

# Earthworm Specific Parameters
[Earthworm]
    MOD_ID = 193
    INST_ID = 6
    HB = 15
    WAVE_RING = 1000
    MLPICK_RING = 1005

# Machine Learning Model
[model]
    obj = 'sbm.EQTransformer()'
    weight_names = ['pnw']
    blinding = (500,500)
    overlap = 3000

# Ayahos Module Initialzation Parameters
[Ayahos]
    module_id = 193
    installation_id = 6
    heartbeat_period = float(15)
    connections = {'WAVE_RING': 1000, 'PICK_RING': 1005}

# Wyrm submodule initialization Parameters
[WaveInWyrm]
    conn_name = 'WAVE_RING'
    max_buffer_length = float(300)

# Seisbench Model TubeWyrm init parameters
[SBMTubeWyrm]
    model = sbm.EQTransformer()
    weight_names = ['pnw']

# Pickout init parameters
[PickOutWyrm]
    conn_name = 'PICK_RING'
    max_buffer_length = float(120)
    buffer_append_method = 1
    prestack_blinding = (500,500)
    leading_mute = 3000
