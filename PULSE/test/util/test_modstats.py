import unittest, pytest
from pandas import Series
from obspy import UTCDateTime
from obspy.core.util.attribdict import AttribDict
from PULSE.util.header import ModStats




class TestModStats(unittest.TestCase):

    def setUp(self):
        self.test_stats = ModStats()
        self.test_starttime = UTCDateTime(0)
        self.test_endtime = UTCDateTime.now()

    def tearDown(self):
        del self.test_stats
        del self.test_starttime
        del self.test_endtime


    def test_init(self):
        self.assertIsInstance(self.test_stats, AttribDict)
        self.assertIsInstance(self.test_stats, ModStats)
        self.assertEqual(self.test_stats.readonly, ['pulserate','runtime'])
        self.assertEqual(self.test_stats._refresh_keys, {'starttime','endtime','niter'})


    def test_setitem(self):
        self.test_stats['starttime'] = self.test_starttime
        self.assertEqual(self.test_stats.starttime, self.test_starttime)
        self.assertEqual(self.test_stats['pulserate'], 0)
        self.assertEqual(self.test_stats['runtime'], 0)
        self.test_stats['niter'] = 2
        self.assertEqual(self.test_stats.niter, 2)
        self.assertEqual(self.test_stats['pulserate'], 0)
        self.assertEqual(self.test_stats['runtime'], 0)
        self.test_stats['endtime'] = self.test_endtime
        self.assertEqual(self.test_stats.endtime, self.test_endtime)
        self.assertGreater(self.test_stats.pulserate, 0)
        self.assertGreater(self.test_stats.runtime, 0)
        for fld in ['pulserate','runtime']:
            with pytest.raises(AttributeError):
                self.test_stats[fld] = 2





# class TestModStats():
#     """Tests for the :class:`~PULSE.data.header.ModStats` class
#     """    
#     def test_class_variables(self):
#         """Test suite for ModStats class variables
#         """        
#         assert ModStats.readonly == ['pulserate','runtime']
#         assert isinstance(ModStats._refresh_keys, set)
#         assert ModStats._refresh_keys == {'starttime','endtime','niter'}
#         assert isinstance(ModStats.defaults, dict)
#         assert isinstance(ModStats._types, dict)

#     def test_init(self):
#         """Test suite for ModStats.__init__
#         """        
#         header = ModStats()
#         assert isinstance(header, AttribDict)
#         # Test _types restrictions
#         for _v in ['a', int(1), float(1.1)]:
#             for _k in ModStats.defaults.keys():
#                 # Catch readonly attribute assignment error
#                 if _k in ModStats.readonly:
#                     with pytest.raises(AttributeError):
#                         ModStats(header={_k:_v})
#                 # Test string attributes
#                 elif _k in ['modname','stop']:
#                     if isinstance(_v, str):
#                         assert ModStats(header={_k:_v})
#                     else:
#                         assert isinstance(ModStats(header={_k:_v})[_k], str)
#                 # Test int attributes
#                 elif _k in ['niter','in0','in1','out0','out1']:
#                     if isinstance(_v, int):
#                         assert ModStats(header={_k:_v})[_k] == _v
#                     elif isinstance(_v, float):
#                         assert ModStats(header={_k:_v})[_k] == int(_v)

#                     else:
#                         with pytest.raises(ValueError):
#                             ModStats(header={_k:_v})
#                 # Test float attributes
#                 else:
#                     if isinstance(_v, float):
#                         assert ModStats(header={_k:_v})[_k] == _v
#                     elif isinstance(_v, int):
#                         assert ModStats(header={_k:_v})[_k] == float(_v)
#                     else:
#                         with pytest.raises(ValueError):
#                             ModStats(header={_k:_v})
    
#     def test_setattr(self):
#         # Test _types restrictions
#         for _v in ['a', int(1), float(1.1)]:
#             for _k in ModStats.defaults.keys():
#                 # SETUP Create new object
#                 header = ModStats()
#                 # Catch readonly attribute assignment error
#                 if _k in ModStats.readonly:
#                     with pytest.raises(AttributeError):
#                         header[_k]= _v
#                 # Test string attributes
#                 elif _k in ['modname','stop']:
#                     # Test string input
#                     if isinstance(_v, str):
#                         header[_k] = _v
#                         assert header[_k] == _v
#                         assert getattr(header, _k) == _v
#                     # Test other input
#                     else:
#                         header[_k] = _v
#                         assert getattr(header, _k) == str(_v)
#                 # Test int attributes
#                 elif _k in ['niter','in0','in1','out0','out1']:
#                     if isinstance(_v, (int, float)):
#                         header[_k] = _v
#                         assert getattr(header, _k) == int(_v)
#                     else:
#                         with pytest.raises(ValueError):
#                             header[_k] = _v
#                 # Test float attributes
#                 else:
#                     if isinstance(_v, (int, float)):
#                         header[_k] = _v
#                         assert getattr(header, _k) == float(_v)
#                     else:
#                         with pytest.raises(ValueError):
#                             header[_k] = _v
#         # Test updates
#         header = ModStats(header={'starttime':0, 'endtime':1, 'niter':3})
#         # Test positive runtime
#         assert header.runtime == 1.
#         assert header.pulserate == 3.
#         # Test longer runtime
#         header.endtime = 2
#         assert header.runtime == 2.
#         assert header.pulserate == 1.5
#         # Test 0 runtime
#         header.endtime = 0
#         assert header.runtime == 0
#         assert header.pulserate == 0
#         # Test negative runtime
#         header.starttime = 1
#         assert header.runtime == -1.
#         assert header.pulserate == 0

#     def test_copy(self):
#         """Test suite for ModStats.copy
#         """        
#         header = ModStats()
#         header2 = header.copy()
#         assert header == header2
#         header2.niter=3
#         header2.endtime=2
#         assert header.niter != header2.niter
#         assert header.endtime != header2.endtime
#         assert header2.runtime == 2.
#         assert header2.pulserate == 1.5
#         assert header.pulserate == 0
#         assert header.runtime == 0
    
#     def test_asdict(self):
#         """Test suite for ModStats.asdict
#         """        
#         header = ModStats()
#         assert isinstance(header.asdict(), dict)
#         hd = header.copy().asdict()
#         for _k, _v in header.items():
#             assert _v == hd[_k]

#     def test_asseries(self):
#         """Test suite for ModStats.asseries
#         """        
#         header = ModStats()
#         ser = header.copy().asseries()
#         assert isinstance(ser, Series)
#         for _ind in header.keys():
#             assert ser[_ind] == header[_ind]