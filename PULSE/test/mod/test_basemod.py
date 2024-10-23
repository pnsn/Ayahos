"""


AI Attribution: Unit test suite developed with assistance from ChatGPT with
and input of the PULSE.mod.base.BaseMod source code. Independently verified
by N. Stevens.
"""

import pytest
from collections import deque

from PULSE.data.header import PulseStats
from PULSE.mod.base import BaseMod

# A fixture to create an instance of BaseMod
@pytest.fixture
def base_mod():
    return BaseMod(max_pulse_size=5, maxlen=10)

def test_initialization_defaults():
    """Test if the BaseMod object is initialized with default values."""
    base_mod = BaseMod()
    assert base_mod.max_pulse_size == 1
    assert base_mod.maxlen is None
    assert isinstance(base_mod.output, deque)
    assert isinstance(base_mod.stats, PulseStats)
    assert base_mod.stats.modname == 'BaseMod'
    assert base_mod._continue_pulsing is True

def test_initialization_with_params():
    """Test BaseMod initialization with custom parameters."""
    base_mod = BaseMod(max_pulse_size=10, maxlen=100, name_suffix='test')
    assert base_mod.max_pulse_size == 10
    assert base_mod.maxlen == 100
    assert '_test' in base_mod.__name__()

def test_invalid_max_pulse_size():
    """Test if ValueError is raised for invalid max_pulse_size."""
    with pytest.raises(ValueError):
        BaseMod(max_pulse_size=0)

def test_invalid_max_pulse_type():
    """Test if TypeError is raised for non-numeric max_pulse_size."""
    with pytest.raises(TypeError):
        BaseMod(max_pulse_size='invalid')

def test_invalid_name_suffix_type():
    """Test if TypeError is raised for invalid name_suffix type."""
    with pytest.raises(TypeError):
        BaseMod(name_suffix={'invalid': 'type'})

def test_repr_and_str(base_mod):
    """Test __repr__ and __str__ methods."""
    assert 'BaseMod' in repr(base_mod)
    assert 'BaseMod' in str(base_mod)

def test_pulse_empty_input(base_mod):
    """Test the pulse method with an empty input deque."""
    input_data = deque()
    output = base_mod.pulse(input_data)
    breakpoint()
    assert len(output) == 0
    assert base_mod.stats.niter == 0

def test_pulse_with_input(base_mod):
    """Test the pulse method with some input data."""
    input_data = deque([1, 2, 3, 4, 5])
    output = base_mod.pulse(input_data)
    assert len(output) == 5
    assert base_mod.stats.niter == 5

def test_deepcopy(base_mod):
    """Test if the deepcopy method works properly."""
    copied_mod = base_mod.copy()
    assert copied_mod is not base_mod
    assert isinstance(copied_mod, BaseMod)

def test_measure_input(base_mod):
    """Test the measure_input method with valid and invalid inputs."""
    input_data = deque([1, 2, 3])
    assert base_mod.measure_input(input_data) == 3

    with pytest.raises(SystemExit):
        base_mod.measure_input('invalid')

def test_measure_output(base_mod):
    """Test the measure_output method."""
    assert base_mod.measure_output() == 0
    base_mod.output.append(1)
    assert base_mod.measure_output() == 1

def test_get_unit_input(base_mod):
    """Test the get_unit_input method."""
    input_data = deque([1, 2, 3])
    unit_input = base_mod.get_unit_input(input_data)
    assert unit_input == 1
    assert len(input_data) == 2

    # Test early stopping with empty input
    empty_data = deque()
    unit_input = base_mod.get_unit_input(empty_data)
    assert unit_input is None
    assert base_mod._continue_pulsing is False

def test_run_unit_process(base_mod):
    """Test the run_unit_process method."""
    unit_input = 42
    unit_output = base_mod.run_unit_process(unit_input)
    assert unit_output == unit_input

def test_store_unit_output(base_mod):
    """Test the store_unit_output method."""
    unit_output = 99
    base_mod.store_unit_output(unit_output)
    assert base_mod.output[-1] == 99

def test_import_class(base_mod):
    """Test the import_class method."""
    cls = base_mod.import_class('collections.deque')
    assert cls is deque

    with pytest.raises(TypeError):
        base_mod.import_class(123)
    
    with pytest.raises(ValueError):
        base_mod.import_class('invalid_class_name')

    with pytest.raises(SystemExit):
        base_mod.import_class('non.existent.Class')



# class TestBaseMod():

#     Logger = logging.getLogger('TestBaseMod')

#     def test_init_bare(self):
#         mod = BaseMod()
#         assert isinstance(mod, object)
#         assert mod.max_pulse_size == 1
#         assert isinstance(mod.output, collections.deque)
#         assert isinstance(mod.Logger, logging.Logger)
#         assert isinstance(mod.maxlen, (type(None), int))
#         assert isinstance(mod.stats, PulseStats)
#         assert mod.stats.modname == 'BaseMod'
#         assert mod._continue_pulsing
#         assert mod.maxlen is None
    
#     def test_init_max_pulse_size(self):
#         mod = BaseMod()
#         assert mod.max_pulse_size == 1
#         mod = BaseMod(max_pulse_size=2)
#         assert mod.max_pulse_size == 2
#         mod = BaseMod(max_pulse_size=1.9)
#         assert mod.max_pulse_size == 1
#         with pytest.raises(ValueError):
#             BaseMod(max_pulse_size=0)
#         with pytest.raises(ValueError):
#             BaseMod(max_pulse_size=-1)
#         with pytest.raises(TypeError):
#             BaseMod(max_pulse_size='a')

#     def test_init_maxlen(self):
#         mod = BaseMod()
#         assert mod.maxlen is None
#         mod = BaseMod(maxlen=2)
#         assert mod.maxlen == 2
#         assert mod.output.maxlen == 2
#         for _i in range(3):
#             mod.output.append(_i)
#             if _i < 1:
#                 assert len(mod.output) == _i + 1
#             else:
#                 assert len(mod.output) == 2
#         with pytest.raises(ValueError):
#             BaseMod(maxlen=-1)

    
#     def test_init_name_suffix(self):
#         mod = BaseMod()
#         assert mod._suffix == ''
#         assert mod.__name__() == 'BaseMod'
#         mod = BaseMod(name_suffix='1')
#         assert mod._suffix == '_1'
#         assert mod.__name__() == 'BaseMod_1'
#         mod = BaseMod(name_suffix=2)
#         assert mod._suffix == '_2'
#         assert mod.__name__() == 'BaseMod_2'
#         with pytest.raises(TypeError):
#             BaseMod(name_suffix = 1.2)
    
#     def test_measure_input(self):
#         mod = BaseMod()
#         input = collections.deque(range(2))
#         assert mod.measure_input(input) == 2


#     def test_get_unit_input(self):
#         mod = BaseMod()
#         input = collections.deque(range(2))
#         assert len(input) == 2
#         assert mod.get_unit_input(input) == 0
#         assert len(input) == 1



#     # def test_pulse(self):
#     #     mod = BaseMod()
#     #     inputs = collections.deque(range(50))

        