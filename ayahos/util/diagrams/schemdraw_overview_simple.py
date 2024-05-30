from schemdraw import flow, Drawing

with Drawing() as D:
    D.config(fontsize=10)
    D.config(unit=1)
    D.config()
    s1 = flow.StateEnd().label('WAVE\nRING')
    flow.Arrow().down(D.unit*5).at(s1.S)

    # Ayahos Module Container
    with D.container() as AY:
        AY.label(
            'Ayahos Module',
            loc='N',
            halign='center',
            valign='top',
            fontsize=14,
            color='firebrick')
        AY.color('firebrick')

        # RingBuffer Submodule
        with AY.container() as RB:
            RB.label('$RingBuffWyrm$ (Submodule)', loc='S', valign='top', halign='center')
            p1 = flow.RoundProcess().label('$RingWyrm$\n"get_wave"')
            flow.Arrow().down(D.unit/2).at(p1.S)
            p2 = flow.RoundProcess().label('BufferWyrm')

        flow.Arrow().right(D.unit*2).at(p2.E)
        # SeisBenchModel TubeWyrm Submodule
        with AY.container() as TW:
            TW.label(
                'SBMTubeWyrm Submodule',
                loc='N',
                halign='center',
                valign='top')
            p3 = flow.RoundProcess().label('$WindowWyrm$\n"PhaseNet"')
            flow.Arrow().length(D.unit*2).theta(60)
            # PreProcessing Submodule
            with TW.container() as PP:
                PP.label('Pre-Processing', loc='N', halign='center', valign='top')
                p4a = flow.RoundProcess().label('$MethodWyrm$\nFill Data Gaps')
                flow.Arrow().down(D.unit/4).at(p4a.S)
                p4b = flow.RoundProcess().label('$MethodWyrm$\nSync Sampling')
                flow.Arrow().down(D.unit/4)
                p4c = flow.RoundProcess().label('$MethodWyrm$\nFill Missing\nChannels')
                flow.Arrow().down(D.unit/4)
                p4d = flow.RoundProcess().label('$MethodWyrm$\nNormalize Traces')
            # Prediction
            flow.Arrow().length(D.unit*2).theta(60).at(p4d.NE)
            p5 = flow.RoundProcess().label('SBMWrym')
        flow.Arrow().right(D.unit*2).at(p5.E)
        # Cluster Picker Submodule
        with AY.container() as CP:
            CP.label('ClustPickWyrm Submodule\n(IN DEVELOPMENT)',
                     loc='S', halign='center', valign='top')
            p6 = flow.RoundProcess().label('$MethodWyrm$\nTrigger')
            flow.Arrow().down(D.unit/2).at(p6.S)
            p7 = flow.RoundProcess().label('$ClusterWyrm$\n(IN DEVELOPMENT)')
        flow.Arrow().right(D.unit*2).at(p7.E)
        p8 = flow.RoundProcess().label('$RingWyrm$\n"put_msg"')
    flow.Arrow().right(D.unit*2)
    s2 = flow.StateEnd().label('PICK\nRING')

        
        