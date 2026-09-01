#!/usr/bin/env python3
"""Minimal single-pass (electron only, old-style) test of B0/Denu to isolate whether the
segfault is pre-existing (VSS_BMIX + MCDecayFinderModule) or caused by the two-pass code
in _generate_and_reco.py."""
import os
import basf2 as b2
import modularAnalysis as ma
from generators import add_evtgen_generator

output = 'output/1_test/isolate_b0denu.root'
os.makedirs(os.path.dirname(output), exist_ok=True)

main = b2.create_path()
main.add_module('EventInfoSetter', evtNumList=3000)
add_evtgen_generator(path=main, finalstate='signal',
                      signaldecfile='decfiles/cocktail/B0/Denu/decay.dec')

decay_str = 'anti-B0 -> e- D+ anti-nu_e ?gamma ?addbrems ...'
ma.findMCDecay('anti-B0:reco', decay_str, appendAllDaughters=True, path=main)
ma.matchMCTruth('anti-B0:reco', path=main)

ma.variablesToNtuple('anti-B0:reco', variables=['mcPDG', 'mcE'],
                      filename=output, treename='reco', path=main)
main.add_module('Progress')
b2.process(main)
print(b2.statistics)
