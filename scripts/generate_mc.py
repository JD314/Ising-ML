"""Generate independent-chain MC data on a (T, h) grid into one HDF5 file.

Example:
    python scripts/generate_mc.py --L 16 --T 1.5 2.269 3.0 --h 0.0 0.1 \
        --chains 8 --samples 500 --burn 1000 --thin 10 --out data/fisher_L16.h5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ising_fisher import dataset, mc  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--T", type=float, nargs="+", required=True)
    p.add_argument("--h", type=float, nargs="+", default=[0.0])
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--samples", type=int, default=500, help="samples per chain")
    p.add_argument("--burn", type=int, default=1000, help="burn-in, in sweeps")
    p.add_argument("--thin", type=int, default=10, help="sweeps between samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for T in args.T:
        for h in args.h:
            # distinct, reproducible seed stream per (T, h)
            seed = hash((args.seed, round(T, 9), round(h, 9))) % (2 ** 32)
            S, E, M, cid = mc.run_chains(args.L, T, h, args.J, args.chains,
                                         args.burn, args.samples, args.thin, seed)
            dataset.save_samples(args.out, args.L, T, h, S, E, M, cid, J=args.J, overwrite=True)
            print(f"saved L={args.L} T={T:.4f} h={h:+.4f}  n={len(E)}")


if __name__ == "__main__":
    main()
