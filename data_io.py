"""Single source of truth for loading the STEREO behaviour dictionaries.

Every script should load CMT/CTM through here rather than opening the bz2
pickles directly, so the dataset version is set in exactly one place.

    from data_io import load_CMT, load_CTM
    CMT = load_CMT()          # current version (DATA_VERSION)
    CTM = load_CTM('Dec24')   # a specific older version, if ever needed

Returns the dictionary exactly as stored, i.e. predictions live at
    D[cohort][mouse][trial]['merged']['predictions']['smartMerge']
for Apr26/Dec24, and directly at D[...]['merged'] for the archived
Aug24/May24 files (which also use the pre-LUT class order).

Fast cache
----------
The .pkl files are bz2 and take ~175 s each to decompress. `build_cache`
writes an uncompressed protocol-5 copy under data/cache/, which loads in a
fraction of the time off SSD. The cache is optional: if it is missing or
stale the bz2 original is read instead. Rebuild with

    python tools/build_data_cache.py

Note: these pickles are the lab's own acquisition output, not untrusted
input, so unpickling them is expected.
"""

import bz2
import os
import pickle

import numpy as np

# --- The dataset version every script uses unless it asks for another ------
DATA_VERSION = 'Apr26'

# Canonical 9-class order, as stored in Apr26/Dec24 smartMerge:
#   0 Jump, 1 Undefined, 2 Floor licking, 3 Wall licking, 4 Grooming,
#   5 Body licking, 6 Rearing, 7 Locomotion, 8 Stationary
CANONICAL_VERSIONS = ('Apr26', 'Dec24')

# Maps the pre-Dec24 class order (Aug24/May24) onto the canonical one.
LUT_OLD_TO_CANONICAL = np.array([4, 5, 3, 2, 6, 1, 8, 7, 0])

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')


def source_path(kind, version=None):
    """Absolute path to the bz2 pickle for `kind` ('CMT'/'CTM')."""
    version = version or DATA_VERSION
    fname = f'{kind}_{version}.pkl'
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, 'archive', fname)
    return path


def cache_path(kind, version=None):
    version = version or DATA_VERSION
    return os.path.join(CACHE_DIR, f'{kind}_{version}.pkl5')


def _cache_is_usable(kind, version):
    """Cache counts as usable only if it is newer than the bz2 it came from."""
    cp, sp = cache_path(kind, version), source_path(kind, version)
    if not os.path.exists(cp):
        return False
    if os.path.exists(sp) and os.path.getmtime(cp) < os.path.getmtime(sp):
        return False
    return True


def load_dict(kind, version=None, use_cache=True, verbose=True):
    """Load CMT or CTM. Prefers the uncompressed cache when it is present."""
    version = version or DATA_VERSION
    if use_cache and _cache_is_usable(kind, version):
        with open(cache_path(kind, version), 'rb') as f:
            D = pickle.load(f)
        if verbose:
            print(f'*** {kind}_{version} loaded from cache ***')
        return D

    path = source_path(kind, version)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{kind}_{version}.pkl not found in {DATA_DIR} or {DATA_DIR}\\archive')
    with bz2.BZ2File(path, 'rb') as f:
        D = pickle.load(f)
    if verbose:
        print(f'*** {kind}_{version} loaded from {os.path.basename(path)} ***')
    return D


CANONICAL_BEHAVIORS = ['Jump', 'Undefined', 'Floor licking', 'Wall licking',
                       'Grooming', 'Body licking', 'Rearing', 'Locomotion',
                       'Stationary']

# Names older scripts used for the same class.
_BEHAVIOR_ALIASES = {
    'back to camera': 'Undefined',
    'undefined': 'Undefined',
}


def relabel_lut(target_order):
    """LUT mapping canonical class indices onto an arbitrary behaviour ordering.

    Several older scripts index behaviours by position in their own `behaviors`
    list, which is not the canonical order (and differs between scripts). Rather
    than hand-maintaining a permutation per script, derive it from the names:

        lut = relabel_lut(behaviors)   # behaviors = that script's own list
        preds = lut[canonical_preds]

    Raises if a canonical behaviour is missing from `target_order`, so a typo
    fails loudly instead of silently mislabelling.
    """
    pos = {}
    for i, name in enumerate(target_order):
        key = _BEHAVIOR_ALIASES.get(name.strip().lower(), name.strip())
        pos[key.lower()] = i
    lut = np.empty(len(CANONICAL_BEHAVIORS), dtype=np.int64)
    missing = []
    for canon_i, name in enumerate(CANONICAL_BEHAVIORS):
        i = pos.get(name.lower())
        if i is None:
            missing.append(name)
        else:
            lut[canon_i] = i
    if missing:
        raise ValueError(
            f'behaviour(s) {missing} not found in target order {list(target_order)}')
    return lut


# Project behaviour palette, in canonical order (see CLAUDE.md).
CANONICAL_COLORS = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7",
                    "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]


def palette_for(target_order):
    """The project palette permuted into an arbitrary behaviour ordering.

    Scripts that index behaviours by position in their own `behaviors` list
    need their colour list in that same order. Rather than hand-maintaining a
    palette per script, permute the canonical one by name:

        colors = data_io.palette_for(behaviors)

    Raises on an unknown behaviour name, so a typo fails loudly rather than
    silently colouring the wrong behaviour.
    """
    lut = relabel_lut(target_order)          # canonical index -> position here
    colors = [None] * len(target_order)
    for canon_i, target_i in enumerate(lut):
        colors[int(target_i)] = CANONICAL_COLORS[canon_i]
    if any(c is None for c in colors):
        unfilled = [target_order[i] for i, c in enumerate(colors) if c is None]
        raise ValueError(f'no canonical colour for {unfilled}')
    return colors


def get_preds(entry, version=None):
    """The frame-by-frame behaviour array for one trial entry.

    Apr26/Dec24 nest it under merged.predictions.smartMerge; the archived
    Aug24/May24 files store it directly at entry['merged'].
    """
    version = version or DATA_VERSION
    if version in CANONICAL_VERSIONS:
        return entry['merged']['predictions']['smartMerge']
    return entry['merged']


def flatten_merged(D, kind='CMT', version=None, remap=True):
    """In place, set entry['merged'] to the canonical prediction array.

    Lets a script say `entry['merged']` and get a 1-D canonical label array
    whatever the underlying dataset version. Every other key (photom, topcam,
    SSD, cam1, cam2) is left untouched.
    """
    version = version or DATA_VERSION
    for lvl1 in D.values():
        for lvl2 in lvl1.values():
            for key, entry in lvl2.items():
                preds = get_preds(entry, version)
                if remap and version not in CANONICAL_VERSIONS:
                    preds = LUT_OLD_TO_CANONICAL[preds]
                entry['merged'] = preds
    return D


def load_CMT(version=None, use_cache=True, verbose=True, flatten=False, remap=True):
    """Cohort -> Mouse -> Trial dictionary.

    flatten=True normalises entry['merged'] to the canonical prediction array.
    """
    D = load_dict('CMT', version, use_cache, verbose)
    return flatten_merged(D, 'CMT', version, remap) if flatten else D


def load_CTM(version=None, use_cache=True, verbose=True, flatten=False, remap=True):
    """Cohort -> Trial -> Mouse dictionary.

    flatten=True normalises entry['merged'] to the canonical prediction array.
    """
    D = load_dict('CTM', version, use_cache, verbose)
    return flatten_merged(D, 'CTM', version, remap) if flatten else D


def build_cache(kind, version=None, verbose=True):
    """Write the uncompressed protocol-5 cache for one dictionary."""
    version = version or DATA_VERSION
    os.makedirs(CACHE_DIR, exist_ok=True)
    D = load_dict(kind, version, use_cache=False, verbose=verbose)
    out = cache_path(kind, version)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(D, f, protocol=5)
    os.replace(tmp, out)
    if verbose:
        print(f'    cache written: {out}  ({os.path.getsize(out)/1e9:.2f} GB)')
    return out
