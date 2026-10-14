# Semileptonic gap from positivity

Constraints on the gap between the inclusive B → X_c ℓν rate and the sum of exclusive modes (SEM), from the
zero-threshold HQE moments of M_X², E_ℓ and q² and the requirement that the gap has a positive spectrum.
The write-up for collaborators is `paper/narrative.tex`.

## Pipeline

| Step | Script | Does | Writes |
|------|--------|------|--------|
| 1 | `1_generate.py` | signal MC (EvtGen, basf2) | `output/1` |
| 2 | `2_reweight.py` | cocktail and gap-mode reweighting (Hammer form factors, MCAmbulance, HFLAV BFs) | `output/2` |
| 3 | `3_sem_modes.py` | moments per SEM mode and its lineshape variants | `data/sem_component_moments.json` |
| 4 | `4_toys.py` | threshold-0 toys (SEM stat, form factors, BFs, gap normalisation) and the bundle | `output/4/gap_toys_bundle.npz` |
| 5 | `5_closure.py` | closure test: inject known gaps, reconstruct, bands per uncertainty source | `output/5`, `figures/5` |
| 6 | `6_data_inversion.py` | MaxEnt spectra of the inclusive moments and of the gap, uncertainty budget | `output/6`, `figures/6` |
| 7 | `7_tests.py` | Results 1–3: shape tests, CDF bounds, support tests, minimum gap size, lineshape variants | `output/7`, `figures/7` |
| 8 | `8_paper.py` | paper figures and every quoted number as TeX macros | `figures/8`, `paper/numbers.tex` |

Steps 5–8 read only the step-4 bundle (plus `output/2` for the candidate modes in step 8). Shared code is in `lib/`:
`gap.py` (one pairing of HQE-fit toys to the SEM toys, used by steps 6–8), `variants.py` (SEM variants),
`cone.py` (positivity test). `inputs/digitize.py` produced the D_sK and Λc p̄ spectra in `inputs/digitized/` once.

## Running (KEKCC)

```bash
source /cvmfs/belle.cern.ch/tools/b2setup && b2setup light-2509-fornax   # step 1: release-09-00-11
python3 3_sem_modes.py --phsp-cocktail FILE
python3 4_toys.py --submit          # toy chunks, merge, bundle
python3 5_closure.py --submit
python3 6_data_inversion.py --submit
python3 7_tests.py --submit
python3 8_paper.py --submit         # after 5-7
cd paper && latexmk -pdf narrative.tex
```

Every `--submit` uses the LSF queue `batch.queue` in `config.yaml`. `output/` is a symlink to `/ghi` and is not in git;
files there migrate to tape (`ghils FILE`: H tape only, B disk and tape, G disk only). Reading an H file or
overwriting a B file can block a job indefinitely, so move old outputs aside (`mv` is safe) before a rerun.
