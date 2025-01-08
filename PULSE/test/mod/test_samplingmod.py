from unittest import TestCase
from pathlib import Path
from collections import deque

from obspy import read

from PULSE.util.header import MLStats
from PULSE.data.dictstream import DictStream
from PULSE.mod.base import BaseMod
from PULSE.mod.sampling import SamplingMod
from PULSE.test.example_data import load_seattle_example

class Test_SamplingMod(TestCase):
    st, _, _ = load_seattle_example(Path().cwd())

    def setUp(self):
        self.ds = DictStream(self.st.copy())
        self.instruments = self.ds.split_on().keys()
        self.mod = SamplingMod(length=2, step=1)
    
    def tearDown(self):
        del self.ds
        del self.instruments
        del self.mod

    def test___init___default(self):
        mod = SamplingMod()
        # Types
        self.assertIsInstance(mod, BaseMod)
        self.assertIsInstance(mod, SamplingMod)
        self.assertIsInstance(mod.output, deque)
        self.assertEqual(mod._input_types, [DictStream])
        self.assertIsInstance(mod.index, dict)
        # Default Values
        self.assertEqual(mod.length, 60.)
        self.assertEqual(mod.step, 42.)
        self.assertEqual(mod.delay, 0.)
        self.assertEqual(mod.min_valid_frac, 0.9)
        self.assertEqual(mod.ref_val, set('Z'))
        self.assertEqual(mod.ref_key,'component')
        self.assertEqual(mod.split_key, 'instrument')
        self.assertFalse(mod.eager)
        self.assertFalse(mod.blind)
        self.assertEqual(mod.stats.mps, 1)
        self.assertEqual(mod.name,'SamplingMod')

    def test___setattr__(self):
        mod = SamplingMod()
        # FloatTypes
        for _key in ['length','step','delay','min_valid_frac','fold_thr']:
            # Type
            with self.assertRaises(TypeError):
                mod.__setattr__(_key, 'a')
            # Lower Limits
            if _key != 'delay':
                with self.assertRaises(ValueError):
                    mod.__setattr__(_key, 0.)
            with self.assertRaises(ValueError):
                mod.__setattr__(_key, -1)
        # Upper Limits
        with self.assertRaises(ValueError):
            mod.__setattr__('min_valid_frac', 1.1)
        with self.assertRaises(ValueError):
            mod.__setattr__('length',10.)
            mod.__setattr__('step', 11.)
        for _key in ['length','delay']:
            with self.assertRaises(NotImplementedError):
                mod.__setattr__(_key, 1e7)
        # ID Keys            
        for _key in ['ref_key','split_key']:
            for _val in dict(MLStats().id_keys).keys():
                mod.__setattr__(_key, _val)
                self.assertEqual(getattr(mod, _key), _val)
            for _val in ['abcd', 1]:
                with self.assertRaises(KeyError):
                    mod.__setattr__(_key, _val)
        # REF VAL
        for _val in ['Z','Z3', set('Z3'), ['UW','OU']]:
            mod.__setattr__('ref_val', _val)
            self.assertEqual(getattr(mod, 'ref_val'), set(_val))
        
        # FIXME
        # for _val in [1, 1., [1,2,3]]:
        #     with self.assertRaises(TypeError):
        #         self.ref_val = _val


        # BOOL ATTR
        for _key in ['eager','blind']:
            for _val in [True, False]:
                mod.__setattr__(_key, _val)
                self.assertEqual(getattr(mod, _key), _val)
            with self.assertRaises(TypeError):
                mod.__setattr__(_key, 'a')
        
    def test__assess_new_entry(self):
        for split_key in dict(MLStats().id_keys).keys():
            mod = SamplingMod(split_key=split_key)
            split_ds = self.ds.split_on(id_key=mod.split_key)
            # Assert that index is empty
            self.assertEqual(len(mod.index), 0)
            # Iterate across split entries
            for _val, _ds in split_ds.items():
                # Ensure dictstream has traces
                self.assertGreater(len(_ds), 0)
                # Assert that al traces share one split_key value
                for _ft in _ds:
                    self.assertEqual(_ft.id_keys[mod.split_key], _val)
                self.assertFalse(_val in mod.index.keys())

                ## RUN PROCESS ##
                mod._assess_new_entry(_val, _ds)
                #################

                # If there are no matches to ref_val
                if not any(_ft.id_keys[mod.ref_key] in mod.ref_val for _ft in _ds):
                    # Assert that there is not a new entry generated
                    self.assertFalse(_val in mod.index.keys())
                # If there are matches to ref_val
                else:
                    self.assertTrue(_val in mod.index.keys())
                    # Assert that ready is False
                    self.assertFalse(mod.index[_val]['ready'])
                    # Assert that all traces are registered
                    self.assertEqual(_ds.traces.keys(), mod.index[_val]['p_ids'].union(mod.index[_val]['s_ids']))
                    self.assertEqual(mod.index[_val]['tf'], mod.index[_val]['ti'] + mod.length)
                    # Assert that primary and secondary ids are correctly registered
                    for _k, _ft in _ds.traces.items():
                        _idk = _ft.id_keys[mod.ref_key]
                        if _idk in mod.ref_val:
                            self.assertIn(_k, mod.index[_val]['p_ids'])
                            # Assert that ti matches the starttime of the primary trace
                            if len(mod.index[_val]['p_ids']) == 1:
                                self.assertEqual(_ft.stats.starttime, mod.index[_val]['ti'])
                            # If there is more than one primary trace, assert that one of them provide the starttime
                            else:
                                results = []
                                for _pid in mod.index[_val]['p_ids']:
                                    results.append(_ds[_pid].stats.starttime == mod.index[_val]['ti'])
                                self.assertTrue(any(results))
                        else:
                            self.assertIn(_k, mod.index[_val]['s_ids'])
    
    def test__assess_entry_readiness(self):
        # Initial tests with diverse data
        for split_key in dict(MLStats().id_keys).keys():
            mod = SamplingMod(split_key=split_key, length=10, step=1)
            split_ds = self.ds.split_on(id_key=mod.split_key)
            for _val, _ds in split_ds.items():
                # Ensure that there are enough data to window
                for _ft in _ds:
                    self.assertGreaterEqual(_ft.count()*_ft.stats.delta, mod.length)
                # Populate index
                mod._assess_new_entry(_val, _ds)
                # If this subset shows up in index
                if _val in mod.index.keys():
                    # Assert that the initial readiness is false
                    self.assertFalse(mod.index[_val]['ready'])
                    # Then assert that following assessment readiness s true
                    self.assertTrue(mod._assess_entry_readiness(_val, _ds))              
        # Setup
        ds = DictStream(read())
        dds = ds.split_on()
        mod_n = self.mod
        mod_n.delay = 10.
        mod_e = SamplingMod(length=2,
                            step=1,
                            eager=True)
        for _k, _v in dds.items():
            mod_n._assess_new_entry(_k, _v)
            mod_e._assess_new_entry(_k, _v)
        # Assert initial works
        self.assertTrue(mod_n._assess_entry_readiness(_k, _v))
        self.assertTrue(mod_e._assess_entry_readiness(_k, _v))
        # Shift index times to run into delay
        mod_n.index[_k]['ti']=ds[0].stats.endtime - 3
        mod_n.index[_k]['tf']=mod_n.index[_k]['ti'] + mod_n.length
        mod_e.index[_k]['ti']=ds[0].stats.endtime - 0.95*mod_e.length
        mod_e.index[_k]['tf']=mod_e.index[_k]['ti'] + mod_e.length
        # Normal + Delay should not result in a true
        self.assertFalse(mod_n._assess_entry_readiness(_k, _v))
        self.assertFalse(mod_n.index[_k]['ready'])
        # Eager should result in a true with small (5%) over-run
        self.assertTrue(mod_e._assess_entry_readiness(_k, _v))
        # Turn off eager
        mod_e.eager = False
        self.assertFalse(mod_e._assess_entry_readiness(_k, _v))
        mod_e.eager = True
        # valid fraction failure should result in false
        mod_e.index[_k]['ti'] += 1
        mod_e.index[_k]['tf'] += 1
        self.assertFalse(mod_e._assess_entry_readiness(_k, _v))

    def test_get_unit_input(self):
        mod = self.mod
        unit_input = mod.get_unit_input(self.ds)
        self.assertIsInstance(unit_input, dict)
        for _k, _v in unit_input.items():
            self.assertIsInstance(_k, str)
            self.assertIsInstance(_v, DictStream)
            # Assert all traces have the same split_key value
            self.assertTrue(all(_ft.id_keys[mod.split_key] == _k for _ft in _v))
            # Assert all traces show readiness
            self.assertTrue(mod.index[_k]['ready'])
        
        # Assert diffset of keys in unit_input are all False readiness
        result = [mod.index[_k]['ready'] for _k in set(mod.index.keys()).difference(unit_input.keys())]
        self.assertTrue(not any(result))

    def test_run_unit_process(self):
        ids = DictStream(read())
        unit_input = self.mod.get_unit_input(ids)
        unit_output = self.mod.run_unit_process(unit_input)
        # Assert unit_output type
        self.assertIsInstance(unit_output, deque)
        # Assert that unit_output has the same length as unit_input
        self.assertEqual(len(unit_input), len(unit_output))
        for _e in unit_output:
            self.assertIsInstance(_e, DictStream)
            breakpoint()
        #     for _ft in _e:

        #         # self.assertLessEqual()
        # for _k, _v in unit_output:
        # breakpoint()


    # def test_run_unit_process(self):
    #     mod = SamplingMod(length=2, step=1)
    #     unit_input = mod.get_unit_input(self.ds)
    #     breakpoint()
