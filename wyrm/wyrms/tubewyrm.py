"""
:module: wyrm.wyrms.tubewyrm
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: AGPL-3.0
:purpose:
    This module defines the Tubewyrm Base Class that facilitates 
    chained execution of pulse(x) class methods for a sequence 
    wyrm objects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input. 
    This `wyrm_queue` is a double ended queue (collections.deque), 
    which provides easier append/pop syntax for editing the wyrm_queue.

    Convenience methods for appending and popping entries from the processing
    queue are provided via behaviors of collections.deque objects
"""
from collections import deque
from wyrm.wyrms.wyrm import Wyrm


class TubeWyrm(Wyrm):
    """
    Base Class facilitating chained execution of pulse(x) class methods
    for a sequence wyrm objects, with each wyrm.pulse(x) taking the prior
    member's pulse(x) output as its input. 
    This `wyrm_queue` is a double ended queue (collections.deque), 
    which provides easier append/pop syntax for editing the wyrm_queue.

    Convenience methods for appending and popping entries from the processing
    queue are provided
    """

    def __init__(self, wyrm_queue=deque([])):
        """
        Create a tubewyrm object
        :: INPUT ::
        :param wyrm_list: [deque] or [list]
                            double ended queue of Wyrm objects
                            if list is provided, will be automatically
                            converted into a deque

        :: OUTPUT ::
        Initialized TubeWyrm object
        """
        # Run compatability checks on wyrm_list
        # If given a single Wyrm, wrap it in a deque
        if isinstance(wyrm_queue, Wyrm):
            self.wyrm_queue = deque([wyrm_queue])
        # If given a list of candidate wyrms, ensure they are all of Wyrm class
        elif isinstance(wyrm_queue, (list, deque)):
            if any(not isinstance(_wyrm, Wyrm) for _wyrm in wyrm_queue):
                print('Not all entries of wyrm_queue are type <class Wyrm>')
                raise TypeError
            # If all members are Wyrms, write to attribute
            else:
                self.wyrm_queue = wyrm_queue
            # Final check that the wyrm_queue is a deque
            if isinstance(wyrm_queue, list):
                self.wyrm_queue = deque(wyrm_queue)
                
        # In any other case:
        else:
            print('Provided wyrm_list was not a list or a Wyrm')
            raise TypeError

    def __repr__(self):
        rstr = '--- Tube ---\n'
        for _i, _wyrm in enumerate(self.wyrm_queue):
            if _i == 0:
                rstr += '(head) '
            else:
                rstr += '       '
            rstr += f'{type(_wyrm)}'
            if _i == len(self.wyrm_queue) - 1:
                rstr += ' (tail)'
            rstr += '\n'

    def append(self, object, end='right'):
        """
        Convenience method for left/right append
        to wyrm_queue

        :: INPUTS ::
        :param object: [Wyrm] candidate wyrm object
        :param end: [str] append side 'left' or 'right'

        :: OUTPUT ::
        None
        """
        if isinstance(object, Wyrm):
            if end.lower() in ['right','r']:
                self.wyrm_list.append(object)
            elif end.lower() in ['left', 'l']
                self.wyrm_queue.appendleft(object)
        
        if isinstance(object, (list, deque)):
            if all(isinstance(_x, Wyrm) for _x in object):
                if end.lower() in ['right', 'r']:
                    self.wyrm_list += deque(object)
                elif end.lower() in ['left', 'l']:
                    self.wyrm_list = deque(object) + self.wyrm_list

        
    def pop(self, end='right'):
        """
        Convenience method for left/right pop
        from wyrm_queue

        :: INPUT ::
        :param end: [str] 'left' or 'right'
        
        :: OUTPUT ::
        :param x: [Wyrm] popped Wyrm object from
                wyrm_queue
        """
        if end.lower() in ['right', 'r']:
            x = self.wyrm_list.pop()
        elif end.lower() in ['left','l']:
            x = self.wyrm_list.popleft()
        return x
    
    def pulse(self, x):
        """
        Initiate a chained pulse for elements of 
        wyrm_queue. 

        E.g., 
        tubewyrm.wyrm_queue = [<wyrm1>, <wyrm2>, <wyrm3>]
        y = tubewyrm.pulse(x) 
            is equivalent to 
        y = wyrm3.pulse(wyrm2.pulse(wyrm1.pulse(x)))

        :: INPUT ::
        :param x: Input `x` for the first Wyrm object in wyrm_queue
        
        :: OUTPUT ::
        :param y: Output `y` from the last Wyrm object in wyrm_queue 
        """
        for _wyrm in self.wyrm_list:
            x = _wyrm.pulse(x)
        y = x
        return y