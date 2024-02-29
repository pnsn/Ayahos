import logger, os, sys, glob
import seisbench.models as sbm
sys.path.append(os.path.join('..'))
from wyrm.util.seisbench import update_windowing_params
import wyrm.core.submodule as wsub
import wyrm.core.io as wio

def prep_models():
    """
    Conduct checks on pretrained weight loadability and local availability for models
    specified in this function. Return a dictionary containing an initiaized model
    object and list of weight names for each model

    :: INPUTS ::
    None

    :: OUTPUTS ::
    :return mod_dict: [dict] composed of [dict] entries with the followign structure
                    mod={
                        'Model0': {'model': model_object, 
                                   'weights': ['wgt_name0','wgt_name1',...]}
                        }
    
    """
    eqt_mod = sbm.EQTransformer()
    test_weights = ['pnw', 'instance', 'stead', 'ethz', 'geofon', 'iquique', 'scedc', 'neic']
    eqt_weights = []
    for _wn in test_weights:
        try:
            eqt_mod.from_pretrained(_wn)
            eqt_weights.append(_wn)
        except ValueError:
            pass
    eqt_mod = update_windowing_params(eqt_mod, overlap=1800, blinding=500)
    mod_dict = {'EQT': {'model': eqt_mod, 'weights': eqt_weights}}


    pn_mod = sbm.PhaseNet()
    test_weights = ['diting','instance','stead','ethz','geofon','iquique','scedc','neic']
    pn_weights = []
    for _wn in test_weights:
        try:
            pn_mod.from_pretrained(_wn)
            pn_weights.append(_wn)
        except ValueError:
            pass
    pn_mod = update_windowing_params(pn_mod, overlap=900, blinding=250)
    mod_dict.update({'PN': {'model': pn_mod, 'weights': pn_weights}})
    
    return mod_dict

def prep_event_file_dict(root_dir, event_glob_str='uw*', file_glob_str='*.mseed'):
    event_dict = {}
    glob_dirs = glob.glob(os.path.join(root_dir, event_glob_str))
    for _d in glob_dirs:
        glob_files = glob.glob(os.path.join(_d, file_glob_str))
        glob_files.sort()
        event_dict.update({os.path.split(_d)[-1]: glob_files})
    return event_dict


def initialize_wyrms(debug=False):
    ### INITIALIZE INPUT 
    disk_d = wio.DiskWyrm(
        event_files=prep_event_file_dict(),
        max_length=300,
        reinit_period=1,
        max_pulse_size=1,
        debug=debug)

    ### INITIALIZE CORE PROCESSING SUBMODULES ###
    # Initialize ML TubeWyrms
    mods = prep_models()
    ml_tube_d_dict = {}
    for _k, _v in mods.items():
        _tube_d = wsub.ML_TubeWyrm(model=_v['model'],
                                   weight_names=_v['weights'],
                                   ml_kwargs={'devicetype':'cpu'})
        ml_tube_d_dict.update({_k: _tube_d})

    # Compose CanWyrm
    ml_can_d = wcoo.CanWyrm(wyrm_dict = ml_tube_d_dict, wait_sec=0, max_pulse_size=1)

    ### INITIAIZE DISTILLATION SUBMODULES ###
    
    spick_d = wsub.SemblancePickerWyrm(skwargs={}, pkwargs={})
    
    ### INITIALIZE OUTPUT SUBMODULE ###
    # msg_type 10 == TYPE_PICK2K - P-wave arriavl time (with 4 digit year) from pick ew
        # NOTE: This may have unintended implications for subsequent processing steps.
        # Might consider using a dedicated memory ring (PICK_ML?) for ML outputs and
        # stick with using the TYPE_PICK2K message type....
    # ALTERNATE IDEA: 
    # msg_type 200 -> Unused in PNSN earthworm.d -> could also use this, but would need to
    # have subsequent parts of EW/AQMS listen for msg_type 200...

    put_d = wio.RingWyrm(module=heart_d.module, conn_id=1, pulse_method_str='put_msg', msg_type=10, max_pulse_size=700*3*10)

    ### COMPOSE MODULE ###
    submod_dict = {'Fetch': disk_d, 'Process': ml_can_d, 'Merge': semb_d, 'Pick': pick_d, 'Submit': put_d}
    heart_d.update(submod_dict)