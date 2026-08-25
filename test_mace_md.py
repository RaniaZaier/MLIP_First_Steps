"""
Comparative study: pristine Si vs Si with an interstitial Na ion,
using the pre-trained MACE-MP-0 potential.

Each structure is first geometry-relaxed (BFGS) to remove unphysical
starting artifacts, then a short Langevin MD run is performed.

Motivation: a minimal analogy for ion-induced local distortion/pressure
in a confined solid matrix -- conceptually related to ion-generated
pressure in porous materials (e.g. alkali-silica reaction).
"""

from mace.calculators import mace_mp
from ase import build, units, Atom
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.io import write
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# --- Parameters ---
TEMPERATURE_K = 300
TIMESTEP_FS = 1
N_STEPS = 100
FRICTION = 0.01
FMAX = 0.05  # relaxation convergence criterion (eV/Å)

# --- Shared MACE calculator (loaded once, reused for both structures) ---
calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu')


def build_pristine():
    """Pristine Si supercell, ~64 atoms."""
    return build.bulk('Si', cubic=True) * (2, 2, 2)


def build_doped(pristine_atoms):
    """Same supercell + one interstitial Na atom near the center."""
    atoms = pristine_atoms.copy()
    cell_center = atoms.get_cell().sum(axis=0) / 2
    na_position = cell_center + np.array([0.7, 0.7, 0.7])
    atoms.append(Atom('Na', position=na_position))
    return atoms


def relax(atoms, label):
    """Geometry relaxation (BFGS) before MD, to remove bad initial contacts."""
    atoms.calc = calc
    e_before = atoms.get_potential_energy()
    opt = BFGS(atoms, logfile=f"{label}_relax.log")
    opt.run(fmax=FMAX)
    e_after = atoms.get_potential_energy()
    print(f"[{label}] Relaxation : {e_before:.4f} eV -> {e_after:.4f} eV")
    return atoms


def run_md(atoms, label):
    """Run a short Langevin MD run, tracking energy and saving structures."""
    atoms.calc = calc

    write(f"{label}_initial.png", atoms)

    e0 = atoms.get_potential_energy()
    print(f"[{label}] Énergie initiale (post-relaxation) : {e0:.4f} eV")

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
    dyn = Langevin(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
    )

    energies = []

    def record_and_print():
        e = atoms.get_potential_energy()
        energies.append(e)
        if dyn.nsteps % 10 == 0:
            print(f"[{label}] Step {dyn.nsteps}: E_pot = {e:.4f} eV")

    dyn.attach(record_and_print, interval=1)
    dyn.run(N_STEPS)

    write(f"{label}_final.xyz", atoms)
    write(f"{label}_final.png", atoms)
    print(f"[{label}] Terminé — structures et trajectoire sauvegardées.\n")

    return energies


# --- Build, relax, then run MD for both systems ---
pristine = build_pristine()
pristine = relax(pristine, "pristine")
energies_pristine = run_md(pristine, "pristine")

doped = build_doped(build_pristine())  # fresh pristine copy before adding Na
doped = relax(doped, "doped")
energies_doped = run_md(doped, "doped")

# --- Comparative energy plot (overlapped) ---
plt.figure(figsize=(7, 5))
plt.plot(energies_pristine, label="Si pristine")
plt.plot(energies_doped, label="Si + Na interstitial")
plt.xlabel("MD step")
plt.ylabel("Potential energy (eV)")
plt.title(f"MACE-MP-0 MD trajectories ({TEMPERATURE_K} K), post-relaxation")
plt.legend()
plt.tight_layout()
plt.savefig("energy_comparison.png", dpi=150)
print("Graphique comparatif sauvegardé dans energy_comparison.png")

# --- 4-panel structure grid: pristine init/final, doped init/final ---
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
panels = [
    ("pristine_initial.png", "Si pristine — initial"),
    ("pristine_final.png", "Si pristine — final"),
    ("doped_initial.png", "Si + Na — initial"),
    ("doped_final.png", "Si + Na — final"),
]

for ax, (fname, title) in zip(axes.flat, panels):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("structures_grid.png", dpi=150)
print("Grille des 4 structures sauvegardée dans structures_grid.png")