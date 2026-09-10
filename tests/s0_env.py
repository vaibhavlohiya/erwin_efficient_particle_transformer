"""S0 - environment check for the ErwinParTv2 merge.

Confirms that `weaver` resolves to a real installed weaver-core and that the
symbols ErwinParTv2 imports from it exist.

Why this is not just `import weaver`: the HEP-NN workspace contains a sibling
`weaver/` fork directory with no __init__.py. A regular package in site-packages
always beats such a namespace directory, so the fork cannot mask a correct
install -- but when weaver-core is ABSENT (e.g. the system python3.13), the fork
dir is still picked up as an empty namespace package and `import weaver`
succeeds anyway. A naive availability check therefore reports "installed" when
nothing is. Checking __file__ and the actual submodules is what catches it.

Run from the weaver conda env:
    python tests/s0_env.py
"""
import importlib
import os
import sys

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

failures = []


def check(name, ok, detail):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


print(f"python {sys.version.split()[0]}  |  cwd {os.getcwd()}")
print(f"workspace {WORKSPACE}\n")

# --- the shadowing trap -----------------------------------------------------
try:
    import weaver
    wpath = getattr(weaver, "__file__", None)
    wpath = os.path.dirname(wpath) if wpath else "<namespace package, no __file__>"
    shadowed = (
        not getattr(weaver, "__file__", None)
        or os.path.commonpath([os.path.abspath(wpath), WORKSPACE]) == WORKSPACE
    )
    check("weaver is a real weaver-core install (not an empty namespace pkg)",
          not shadowed, wpath)
except ImportError as e:
    check("weaver importable", False, str(e))

# --- the pieces the merge actually needs ------------------------------------
NEEDED = [
    ("weaver.nn.model.ParticleTransformer", ["Block", "Embed", "SequenceTrimmer",
                                             "pairwise_lv_fts", "build_sparse_tensor"]),
    ("weaver.utils.logger", ["_logger"]),
]
for mod_name, attrs in NEEDED:
    try:
        mod = importlib.import_module(mod_name)
        missing = [a for a in attrs if not hasattr(mod, a)]
        check(mod_name, not missing, "all symbols present" if not missing
              else f"missing {missing}")
    except ImportError as e:
        check(mod_name, False, str(e))

for name in ("torch", "numpy", "uproot", "einops", "awkward"):
    try:
        m = importlib.import_module(name)
        check(name, True, getattr(m, "__version__", "?"))
    except ImportError as e:
        check(name, False, str(e))

# --- device -----------------------------------------------------------------
try:
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    check("torch device", True, dev)
    # ErwinParTv2 must never rely on -inf in SDPA masks (NaN on MPS); record it.
    check("torch.nn.functional.scaled_dot_product_attention", True,
          "available" if hasattr(torch.nn.functional,
                                 "scaled_dot_product_attention") else "MISSING")
except ImportError:
    pass

# --- balltree is optional: oracle only, never in the training path ----------
try:
    import balltree  # noqa: F401
    print("  [note] balltree present - S2 parity check can run in this env")
except ImportError:
    print("  [note] balltree absent - run S2 in an env that has balltree-erwin "
          "(training path must never import it)")

# --- the data ---------------------------------------------------------------
root = os.path.join(WORKSPACE, "JetClass_example_100k.root")
local = os.path.join(WORKSPACE, "efficient_particle_transformer",
                     "JetClass_example_100k.root")
found = next((p for p in (local, root) if os.path.exists(p)), None)
check("JetClass_example_100k.root", found is not None,
      f"{found} ({os.path.getsize(found) / 1e6:.0f} MB)" if found else "not found")

print()
if failures:
    print(f"S0 FAILED: {len(failures)} check(s) - {', '.join(failures)}")
    sys.exit(1)
print("S0 PASSED")
