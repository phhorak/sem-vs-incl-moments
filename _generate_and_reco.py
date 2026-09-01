#!/usr/bin/env python3
"""Generate signal MC with EvtGen and reconstruct B -> Xc l nu (l = e, mu) in one step.
One job per mode; mode determines which decay.dec and which findMCDecay strings to run.

Every job runs findMCDecay twice against the same generated sample -- once for the
electron-flavor decay string, once for the muon-flavor one -- and merges the two matched
lists with an extraInfo(decayModeID) tag (0=electron, 1=muon). This is required for the
gap-mode decfiles, which generate a 50/50 e/mu split per event: truth-matching only the
electron half would silently discard the muon half. For single-flavor cocktail decfiles
the non-matching flavor's findMCDecay call is harmless -- it just returns zero candidates.
"""

import argparse
import re
import os
import basf2 as b2
import modularAnalysis as ma
from generators import add_evtgen_generator
from variables import variables as vm

parser = argparse.ArgumentParser()
parser.add_argument('--dec_file', required=True)
parser.add_argument('--output',   required=True)
parser.add_argument('--mode',     required=True, help='e.g. Bplus/Dstenu or B0/Dstenu')
parser.add_argument('--nevents',  type=int, default=100000)
args = parser.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# ── mode table ────────────────────────────────────────────────────────────────
# findMCDecay strings. Keep permissive tail for radiative/additional truth content.
MODES = {
    # B+ modes
    'Bplus/Dstenu':       'B+ -> e+ anti-D*0 nu_e ?gamma ?addbrems ...',
    'Bplus/Denu':         'B+ -> e+ anti-D0 nu_e ?gamma ?addbrems ...',
    'Bplus/D1enu':        'B+ -> e+ anti-D_10 nu_e ?gamma ?addbrems ...',
    'Bplus/D0stenu':      'B+ -> e+ anti-D_0*0 nu_e ?gamma ?addbrems ...',
    "Bplus/Dp1enu":       "B+ -> e+ anti-D'_10 nu_e ?gamma ?addbrems ...",
    'Bplus/D2stenu':      'B+ -> e+ anti-D_2*0 nu_e ?gamma ?addbrems ...',
    'Bplus/Dstmpienu':    'B+ -> e+ D*- pi+ nu_e ?gamma ?addbrems ...',
    'Bplus/Dst0pi0enu':   'B+ -> e+ anti-D*0 pi0 nu_e ?gamma ?addbrems ...',
    'Bplus/Dmpienu':      'B+ -> e+ D- pi+ nu_e ?gamma ?addbrems ...',
    'Bplus/D0pi0enu':     'B+ -> e+ anti-D0 pi0 nu_e ?gamma ?addbrems ...',
    'Bplus/D0pipipenu':   'B+ -> e+ anti-D0 pi+ pi- nu_e ?gamma ?addbrems ...',
    'Bplus/Dmpipizenu':   'B+ -> e+ D- pi+ pi0 nu_e ?gamma ?addbrems ...',
    'Bplus/D0pizpizenu':  'B+ -> e+ anti-D0 pi0 pi0 nu_e ?gamma ?addbrems ...',
    'Bplus/Dst0pipipenu': 'B+ -> e+ anti-D*0 pi+ pi- nu_e ?gamma ?addbrems ...',
    'Bplus/Dstmpipizenu': 'B+ -> e+ D*- pi+ pi0 nu_e ?gamma ?addbrems ...',
    'Bplus/Dst0pizpizenu':'B+ -> e+ anti-D*0 pi0 pi0 nu_e ?gamma ?addbrems ...',
    'Bplus/DsstkKenu':    'B+ -> e+ D_s*- K+ nu_e ?gamma ?addbrems ...',
    'Bplus/DsKenu':       'B+ -> e+ D_s- K+ nu_e ?gamma ?addbrems ...',
    'Bplus/D0etaenu':     'B+ -> e+ anti-D0 eta nu_e ?gamma ?addbrems ...',
    'Bplus/Dst0etaenu':   'B+ -> e+ anti-D*0 eta nu_e ?gamma ?addbrems ...',
    # gap modes (both prefixed and bare keys for flexible submission)
    'gap_modes/Bplus/D2stDeta':  'B+ -> e+ anti-D_2*0 nu_e ?gamma ?addbrems ...',
    'gap_modes/Bplus/DummyDeta': 'B+ -> e+ anti-D_0*0 nu_e ?gamma ?addbrems ...',
    'gap_modes/Bplus/Dp1DstEta': "B+ -> e+ anti-D'_10 nu_e ?gamma ?addbrems ...",
    'gap_modes/Bplus/Dp1Deta':   "B+ -> e+ anti-D'_10 nu_e ?gamma ?addbrems ...",
    'gap_modes/Bplus/DsKenu':    'B+ -> e+ anti-D_0*0 nu_e ?gamma ?addbrems ...',
    'gap_modes/Bplus/LcPenu':         'B+ -> e+ anti-D_0*0 nu_e ?gamma ?addbrems ...',
    'gap_modes/Bplus/D0stDeta':       'B+ -> e+ nu_e ... ?gamma ?addbrems',
    'gap_modes/Bplus/D1_2550Deta':    'B+ -> e+ nu_e ... ?gamma ?addbrems',
    'gap_modes/Bplus/D1Dgamma':       'B+ -> e+ anti-D_10 nu_e ?gamma ?addbrems ...',
    'gap_modes/Bplus/D0pipipipenu':   'B+ -> e+ nu_e ... ?gamma ?addbrems',
    'gap_modes/Bplus/Dst0pipipipenu': 'B+ -> e+ nu_e ... ?gamma ?addbrems',
    'Bplus/D2stDeta':                 'B+ -> e+ anti-D_2*0 nu_e ?gamma ?addbrems ...',
    'Bplus/DummyDeta':           'B+ -> e+ anti-D_0*0 nu_e ?gamma ?addbrems ...',
    'Bplus/Dp1DstEta':           "B+ -> e+ anti-D'_10 nu_e ?gamma ?addbrems ...",
    'Bplus/Dp1Deta':             "B+ -> e+ anti-D'_10 nu_e ?gamma ?addbrems ...",
    'Bplus/LcPenu':              'B+ -> e+ anti-D_0*0 nu_e ?gamma ?addbrems ...',
    # anti-B0 modes
    'B0/Dstenu':          'anti-B0 -> e- D*+ anti-nu_e ?gamma ?addbrems ...',
    'B0/Denu':            'anti-B0 -> e- D+ anti-nu_e ?gamma ?addbrems ...',
    'B0/D1enu':           'anti-B0 -> e- D_1+ anti-nu_e ?gamma ?addbrems ...',
    'B0/D0stenu':         'anti-B0 -> e- D_0*+ anti-nu_e ?gamma ?addbrems ...',
    "B0/Dp1enu":          "anti-B0 -> e- D'_1+ anti-nu_e ?gamma ?addbrems ...",
    'B0/D2stenu':         'anti-B0 -> e- D_2*+ anti-nu_e ?gamma ?addbrems ...',
    'B0/Dst0pienu':       'anti-B0 -> e- D*0 pi+ anti-nu_e ?gamma ?addbrems ...',
    'B0/Dstpi0enu':       'anti-B0 -> e- D*+ pi0 anti-nu_e ?gamma ?addbrems ...',
    'B0/D0pienu':         'anti-B0 -> e- D0 pi+ anti-nu_e ?gamma ?addbrems ...',
    'B0/Dpi0enu':         'anti-B0 -> e- D+ pi0 anti-nu_e ?gamma ?addbrems ...',
    'B0/Dpipipenu':       'anti-B0 -> e- D+ pi+ pi- anti-nu_e ?gamma ?addbrems ...',
    'B0/D0pipipzenu':     'anti-B0 -> e- D0 pi+ pi0 anti-nu_e ?gamma ?addbrems ...',
    'B0/Dpizpizenu':      'anti-B0 -> e- D+ pi0 pi0 anti-nu_e ?gamma ?addbrems ...',
    'B0/Dstpipipenu':     'anti-B0 -> e- D*+ pi+ pi- anti-nu_e ?gamma ?addbrems ...',
    'B0/Dst0pipipzenu':   'anti-B0 -> e- D*0 pi+ pi0 anti-nu_e ?gamma ?addbrems ...',
    'B0/Dstpizpizenu':    'anti-B0 -> e- D*+ pi0 pi0 anti-nu_e ?gamma ?addbrems ...',
    'B0/Detaenu':         'anti-B0 -> e- D+ eta anti-nu_e ?gamma ?addbrems ...',
    'B0/Dstetaenu':       'anti-B0 -> e- D*+ eta anti-nu_e ?gamma ?addbrems ...',
    # B0 gap modes (isospin partners of the Bplus gap modes above; full-isospin generalization)
    'gap_modes/B0/D2stDeta':        'anti-B0 -> e- D_2*+ anti-nu_e ?gamma ?addbrems ...',
    'gap_modes/B0/DummyDeta':       'anti-B0 -> e- D_0*+ anti-nu_e ?gamma ?addbrems ...',
    'gap_modes/B0/Dp1DstEta':       "anti-B0 -> e- D'_1+ anti-nu_e ?gamma ?addbrems ...",
    'gap_modes/B0/Dp1Deta':         "anti-B0 -> e- D'_1+ anti-nu_e ?gamma ?addbrems ...",
    'gap_modes/B0/DsKenu':          'anti-B0 -> e- D_0*+ anti-nu_e ?gamma ?addbrems ...',
    'gap_modes/B0/LcPenu':          'anti-B0 -> e- D_0*+ anti-nu_e ?gamma ?addbrems ...',
    'gap_modes/B0/D0stDeta':        'anti-B0 -> e- anti-nu_e ... ?gamma ?addbrems',
    'gap_modes/B0/D1_2550Deta':     'anti-B0 -> e- anti-nu_e ... ?gamma ?addbrems',
    'gap_modes/B0/D1Dgamma':        'anti-B0 -> e- D_1+ anti-nu_e ?gamma ?addbrems ...',
    'gap_modes/B0/D0pipipipenu':    'anti-B0 -> e- anti-nu_e ... ?gamma ?addbrems',
    'gap_modes/B0/Dst0pipipipenu':  'anti-B0 -> e- anti-nu_e ... ?gamma ?addbrems',
}


def _lepton_swap(decay_str):
    """Convert an electron-flavor findMCDecay string to its muon-flavor equivalent."""
    s = decay_str
    s = re.sub(r'\banti-nu_e\b', 'anti-nu_mu', s)
    s = re.sub(r'(?<![\w-])nu_e\b', 'nu_mu', s)
    s = re.sub(r'(?<![\w-])e\+', 'mu+', s)
    s = re.sub(r'(?<![\w-])e-(?!\w)', 'mu-', s)
    return s


def _family_key(mode):
    """Cocktail muon modes (e.g. 'Bplus/Dmunu') share their electron sibling's MODES entry;
    the muon findMCDecay string is derived from it via _lepton_swap. Gap-mode keys have no
    lepton suffix and pass through unchanged (both flavors already have their own MODES entry
    where needed, or share one bare key for gap modes with an inline 50/50 e/mu decfile)."""
    if mode.endswith('munu'):
        return mode[:-4] + 'enu'
    return mode


lookup_key = _family_key(args.mode)
if lookup_key not in MODES:
    raise ValueError(f'unknown mode {args.mode!r} (resolved to {lookup_key!r})')

decay_str_e  = MODES[lookup_key]
decay_str_mu = _lepton_swap(decay_str_e)
is_Bplus     = 'Bplus' in args.mode
base_particle = 'B+' if is_Bplus else 'anti-B0'
B_list    = f'{base_particle}:reco'
B_list_e  = f'{base_particle}:reco_e'
B_list_mu = f'{base_particle}:reco_mu'

# ── basf2 path ────────────────────────────────────────────────────────────────

main = b2.create_path()
main.add_module('EventInfoSetter', evtNumList=args.nevents)
add_evtgen_generator(path=main, finalstate='signal', signaldecfile=args.dec_file)

# Two independent truth-matching passes over the same generated sample: one per lepton
# flavor. Whichever flavor a given decfile/event doesn't produce simply yields zero
# candidates for that pass -- harmless. decayModeID (0=e, 1=mu) tags provenance downstream.
ma.findMCDecay(B_list_e,  decay_str_e,  appendAllDaughters=True, path=main)
ma.findMCDecay(B_list_mu, decay_str_mu, appendAllDaughters=True, path=main)
ma.matchMCTruth(B_list_e,  path=main)
ma.matchMCTruth(B_list_mu, path=main)
ma.variablesToExtraInfo(B_list_e,  {'constant(0)': 'decayModeID'}, path=main)
ma.variablesToExtraInfo(B_list_mu, {'constant(1)': 'decayModeID'}, path=main)
ma.copyLists(B_list, [B_list_e, B_list_mu], path=main)

# ── variables ─────────────────────────────────────────────────────────────────

vm.addAlias('B_ECM',     'useCMSFrame(E)')
vm.addAlias('B_pCM',     'useCMSFrame(p)')
vm.addAlias('B_thetaCM', 'useCMSFrame(theta)')
vm.addAlias('B_phiCM',   'useCMSFrame(phi)')

# B meson lab-frame components for Hammer (mcE and mcPDG stored directly)
vm.addAlias('B_mcPX', 'mcPX')
vm.addAlias('B_mcPY', 'mcPY')
vm.addAlias('B_mcPZ', 'mcPZ')

for i in range(5):
    vm.addAlias(f'd{i}_E',     f'daughter({i}, useCMSFrame(E))')
    vm.addAlias(f'd{i}_px',    f'daughter({i}, useCMSFrame(px))')
    vm.addAlias(f'd{i}_py',    f'daughter({i}, useCMSFrame(py))')
    vm.addAlias(f'd{i}_pz',    f'daughter({i}, useCMSFrame(pz))')
    vm.addAlias(f'd{i}_mcPDG', f'daughter({i}, mcPDG)')

# Lab-frame 4-vectors for Hammer: B daughters (findMC order i=0: Xc, i=1: lepton, i=2: nu)
for i in range(3):
    vm.addAlias(f'd{i}_mcE',  f'daughter({i}, mcE)')
    vm.addAlias(f'd{i}_mcPX', f'daughter({i}, mcPX)')
    vm.addAlias(f'd{i}_mcPY', f'daughter({i}, mcPY)')
    vm.addAlias(f'd{i}_mcPZ', f'daughter({i}, mcPZ)')

# Keep branch names d1d* for downstream compatibility, but in findMC output
# the Xc resonance sits at daughter(0).
for j in range(2):
    vm.addAlias(f'd1d{j}_mcE',   f'daughter(0, daughter({j}, mcE))')
    vm.addAlias(f'd1d{j}_mcPX',  f'daughter(0, daughter({j}, mcPX))')
    vm.addAlias(f'd1d{j}_mcPY',  f'daughter(0, daughter({j}, mcPY))')
    vm.addAlias(f'd1d{j}_mcPZ',  f'daughter(0, daughter({j}, mcPZ))')
    vm.addAlias(f'd1d{j}_mcPDG', f'daughter(0, daughter({j}, mcPDG))')

# Xc granddaughters (D** -> D* -> daughters; needed for narrow D** modes)
for j in range(2):
    vm.addAlias(f'd1d0d{j}_mcE',   f'daughter(0, daughter(0, daughter({j}, mcE)))')
    vm.addAlias(f'd1d0d{j}_mcPX',  f'daughter(0, daughter(0, daughter({j}, mcPX)))')
    vm.addAlias(f'd1d0d{j}_mcPY',  f'daughter(0, daughter(0, daughter({j}, mcPY)))')
    vm.addAlias(f'd1d0d{j}_mcPZ',  f'daughter(0, daughter(0, daughter({j}, mcPZ)))')
    vm.addAlias(f'd1d0d{j}_mcPDG', f'daughter(0, daughter(0, daughter({j}, mcPDG)))')

daughter_vars = [v for i in range(5)
                 for v in (f'd{i}_E', f'd{i}_px', f'd{i}_py', f'd{i}_pz', f'd{i}_mcPDG')]

hammer_daughter_vars = (
    [f'd{i}_{v}' for i in range(3) for v in ('mcE', 'mcPX', 'mcPY', 'mcPZ')] +
    [f'd1d{j}_{v}' for j in range(2) for v in ('mcE', 'mcPX', 'mcPY', 'mcPZ', 'mcPDG')] +
    [f'd1d0d{j}_{v}' for j in range(2) for v in ('mcE', 'mcPX', 'mcPY', 'mcPZ', 'mcPDG')]
)

variables = [
    'B_ECM', 'B_pCM', 'B_thetaCM', 'B_phiCM',
    'mcPDG', 'mcE', 'mcP',
    'B_mcPX', 'B_mcPY', 'B_mcPZ',
    *daughter_vars,
    *hammer_daughter_vars,
    'eventRandom',
    'extraInfo(decayModeID)',
]

ma.variablesToNtuple(B_list, variables=variables,
                     filename=args.output, treename='reco', path=main)

main.add_module('Progress')
b2.process(main)
print(b2.statistics)
