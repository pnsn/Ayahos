import schemdraw
from schemdraw.flow import *


with schemdraw.Drawing() as d:
    # Primary Configuration
    d.config(fontsize=8)

    ## MAIN STACK ##
    # Initialize startpoint
    s = Start().label('START\nPULSE')
    Arrow().down(d.unit/2)
    # Input Block
    i1 = Data().label(':arg: $input$')
    Arrow().down(d.unit/2)
    # p1 = Process().label('Measure\n$input$')
    # Arrow().down(d.unit/2)

    # Start of For Loop
    d1 = Decision(E='YES', W='NO').label('iterno\n$\leq$\nmax_pulse_size')
    Arrow().right(d.unit/2).at(d1.E)
    d2 = Decision(E='YES', S='NO').label('run this iteration?')
    # Kickright
    Arrow().right(d.unit/2).at(d2.E)
    p2 = Process().label('get `unit_input`\nfrom $input$')
    Arrow().down(d.unit/2).at(p2.S)
    p3 = Process().label('unit process')
    Arrow().down(d.unit/2)
    p4 = Process().label('capture unit output')
    Arrow().left(d.unit*2.5).at(p4.W)
    d3 = Decision(N='YES', W='NO').label('run next iteration')
    # Arrow().down(d.unit/2)
    # p2 = Process().label('')
    # d3 = Decision(S='YES', W='NO').label('run next iteration?')

    # Connections
    Wire().at(s.S).to(i1.N)
    Wire().at(d1.S).to(d2.N)


    # Sta
    d += Start().label('START\nPULSE')
    d += Arrow().down(d.unit/2)
    d += Data().label('input')
    d += Arrow().down(d.unit/2)
    # Measure input outside for-loop
    d += Process().label('measure input')
    d += Arrow().down(d.unit/2)