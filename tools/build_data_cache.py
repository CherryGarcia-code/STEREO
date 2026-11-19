"""Build the uncompressed fast-load cache for the current DATA_VERSION.

    python tools/build_data_cache.py            # current version, CMT + CTM
    python tools/build_data_cache.py Dec24      # a specific version

Each bz2 dictionary takes ~175 s to decompress; the cache it writes loads
in a fraction of that off SSD, which matters when re-running the whole
figure pipeline. Safe to delete at any time -- data_io falls back to the
bz2 originals, and also ignores a cache older than its source file.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_io import DATA_VERSION, build_cache, cache_path, load_dict  # noqa: E402

version = sys.argv[1] if len(sys.argv) > 1 else DATA_VERSION

for kind in ('CMT', 'CTM'):
    t0 = time.time()
    print(f'building cache for {kind}_{version} ...', flush=True)
    build_cache(kind, version)
    print(f'    built in {time.time() - t0:.0f}s', flush=True)

    t0 = time.time()
    D = load_dict(kind, version)
    n = sum(len(v2) for v1 in D.values() for v2 in v1.values())
    print(f'    verify: reload {time.time() - t0:.0f}s, '
          f'{len(D)} cohorts, {n} entries', flush=True)
    del D
