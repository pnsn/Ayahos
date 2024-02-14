from obspy import read
from wyrm.wyrms.sequence import TubeWyrm
from wyrm.wyrms.window import WindowWyrm
from wyrm.wyrms.process import ProcWyrm
from wyrm.structures.rtinststream import RtInstStream
from wyrm.structures.window import InstWindow
import seisbench.models as sbm
import torch

# Compose RtInstStream from disk
st = read('../../example/uw61957912/bulk.mseed')
rtis = RtInstStream(max_length=300.).append(st)

# Initialize SeisBench model
model = sbm.EQTransformer().from_pretrained('pnw')
device = torch.device('mps')

# Initialize windowwyrm
windwyrm = WindowWyrm().populate_from_seisbench(model)
# Initialize procwyrm
procwyrm = ProcWyrm(
    target_class=InstWindow,
    self_eval_string_list=['.wind_split()',
                           '.detrend("demean")',
                           '.detrend("linear")',
                           '.resample(self._target_sr)',
                           '.taper(None, max_length=0.06, side="both")',
                           '.wind_merge()',
                           '.wind_sync()',
                           '.wind_trim()'],
    out_eval_string='.to_torch()')
# Initialize SeisBenchWyrm
