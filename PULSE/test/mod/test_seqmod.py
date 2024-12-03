"""
:module: PULSE.test.mod.test_seqmod
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose: A :class:`~unittest.TestSuite` for :class:`~PULSE.mod.sequence.SeqMod`
"""
import unittest, logging, time
from collections import deque
from pandas import DataFrame
from PULSE.mod.base import BaseMod
from PULSE.mod.sequence import Sequence, SeqMod


class TestSeqMod(unittest.TestCase):
    """A :class:`~unittest.Testcase` for :class:`~.SeqMod`
    """
    def setUp(self):
        """setUp for unit tests - make:
         - an input of integers
         - a sequence of :class:`~.BaseMod` objects with unique suffixes
         - a SeqMod hosting the sequence
        """        
        self.test_input = deque(range(20))
        self.test_seq = Sequence([BaseMod(name=str(_e), max_pulse_size=4-_e) for _e in range(3)])
        self.test_mod = SeqMod(self.test_seq, name='Test')

    def tearDown(self) -> None:
        """tearDown for unit tests - delete all objects created in :meth:`~.TestSeqMod.setUp`
        """        
        del self.test_input
        del self.test_seq
        del self.test_mod


    def test___init__(self):
        """Test suite for :meth:`~.SeqMod.__init__` with :meth:`~.TestSeqMod.setUp` inputs
        """        
        # Check types
        self.assertIsInstance(self.test_mod, BaseMod)
        self.assertIsInstance(self.test_mod, SeqMod)
        self.assertIsInstance(self.test_mod.sequence, Sequence)
        # Check empty output match
        self.assertEqual(self.test_mod.output, self.test_seq['BaseMod_2'].output)
        # Check input_types match
        self.assertEqual(self.test_mod._input_types, self.test_seq['BaseMod_0']._input_types)
        # Check other reasonable inputs
        for modules in [BaseMod(), {BaseMod()}, {'BaseMod': BaseMod()}, (BaseMod())]:
            self.assertIsInstance(SeqMod(modules), SeqMod)

        # ValueError with maxlen
        for maxlen in [None, -1, 0, 1201]:
            with self.assertRaises(ValueError):
                SeqMod(self.test_seq, maxlen=None)
        # Other types for maxlen raises TypeError
        with self.assertRaises(TypeError):
            SeqMod(self.test_seq, maxlen='a')
        
        # Other types for modules raises errors forwarded from Sequence
        with self.assertRaises(TypeError):
            SeqMod(int)
        with self.assertRaises(KeyError):
            SeqMod({'BaseMod_1': BaseMod(name='0')})
        for modules in ([1], {'BaseMod_0': 'abc'}):
            with self.assertRaises(TypeError):
                SeqMod(modules)

    def test_get_unit_input(self):
        """Test suite for :meth:`~.SeqMod.get_unit_input`
        """ 
        # Assert that the unit input is identical to the input
        self.assertEqual(self.test_mod.get_unit_input(self.test_input),
                         self.test_input)

    def test_run_unit_process(self):
        """Test suite for :meth:`~.SeqMod.run_unit_process`
        """        
        # Run unit process
        unit_output = self.test_mod.run_unit_process(self.test_input)
        # Assert that the output is a DataFrame - SeqMod.pulse proudces a summary table of its contents' pulse stats
        self.assertIsInstance(unit_output, DataFrame)
        # Assert that metadata have not updated as of yet (this is handled in put_unit_output)
        self.assertEqual(len(self.test_mod.metadata), 0)
        
    def test_put_unit_output(self):
        """Test suite for :meth:`~.SeqMod.put_unit_output`
        """       
        # Make an example SeqMod that only logs for 1 
        seqmod = SeqMod(modules=self.test_seq, maxlen=1.0)
        # Pulse once
        _ = seqmod.pulse(self.test_input)
        self.assertEqual(len(seqmod.metadata), 3)
        # Wait half of maxlen
        time.sleep(0.5)
        # Pulse again
        _ = seqmod.pulse(self.test_input)
        self.assertEqual(len(seqmod.metadata), 6)
        # Wait other half of maxlen + 10%
        time.sleep(0.6)
        # Pulse again
        _ = seqmod.pulse(self.test_input)
        # And assert that 3-6 entries exist
        self.assertLessEqual(len(seqmod.metadata), 6)
        self.assertGreaterEqual(len(seqmod.metadata), 3)
        # Assert that the ages of all the metadata do not exceed maxlen
        self.assertTrue(all([seqmod.metadata.endtime.max() - row.starttime <= 1. for _, row in seqmod.metadata.iterrows()]))

    def test_pulse(self):
        """Test suite for :meth:`~.SeqMod.pulse`
        """
        # Run a pulse
        _ = self.test_mod.pulse(self.test_input)
        # Assert output is deque 
        self.assertIsInstance(self.test_mod.output, deque)
        # Assert that test_mod.output is identical to the output of the last element
        self.assertEqual(self.test_mod.output, self.test_mod.sequence.last.output)
        # Assert that the metadata attribute is a pandas.DataFrame
        self.assertIsInstance(self.test_mod.metadata, DataFrame)
        # Assert that the metadata attribute is the same length as the sequence
        self.assertEqual(len(self.test_mod.metadata), len(self.test_mod.sequence))
    
    def test_pulse_on_empty(self):
        """Specific test case for :meth:`~.SeqMod.pulse` where an empty input is provided
        """        
        seqmod = SeqMod(modules=Sequence())
        with self.assertLogs(level=logging.CRITICAL):
            seqmod.pulse(self.test_input)


    # def test_    
    


    # def test_initialization(self):
    #     self.assertIsInstance(self.test_mod.sequence, Sequence)
    #     # self.assertEqual(self.test_mod.sequence, {'BaseMod_0': BaseMod(name='0')})
    #     self.assertEqual(self.test_mod.stats.mps, 1)
    # #     # Locked to always None - set by last item in sequence
    # #     self.assertIsNone(self.test_mod.stats.maxlen)
    # #     self.assertEqual(self.test_mod.stats.name, 'SeqMod_Test')
    # #     self.assertIsInstance(self.test_mod.stats)
    # #     self.assertEqual(self.test_mod._max_age, 60.)

    # # def test_invalid_name_type(self):
    #     TestBaseMod(self)

        