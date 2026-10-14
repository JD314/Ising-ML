"""First Stage-1 exploration of the Fisher Schur invariant (CONTEXT_ISING_ML.md §6)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ising_fisher import dataset, fisher, mc  # noqa: E402
from ising_fisher.models import SimpleCNN, SimpleGenCNN, SimpleMLP  # noqa: E402
from ising_fisher.train import (evaluate_classifier, evaluate_generator,  # noqa: E402
                                train_classifier, train_generator)

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--Tmin", type=float, default=1.5)
    p.add_argument("--Tmax", type=float, default=3.2)
    p.add_argument("--nT", type=int, default=8)
    p.add_argument("--chains", type=int, default=8, help="independent chains per T (half train / half eval)")
    p.add_argument("--samples", type=int, default=3000, help="samples per chain")
    p.add_argument("--burn", type=int, default=500, help="burn-in sweeps")
    p.add_argument("--thin", type=int, default=10, help="sweeps between samples")
    p.add_argument("--clf-epochs", type=int, default=15)
    p.add_argument("--gen-epochs", type=int, default=15)
    p.add_argument("--n-model-samples", type=int, default=12000,
                   help="model samples x ~ q(.|T) per T for the generative scores")
    p.add_argument("--rcond", type=float, default=1e-6)
    p.add_argument("--clf-rcond", type=float, default=1e-3,
                   help="SVD cutoff for classifier r_f to avoid low-T sector overfitting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=None, help="HDF5 cache (default data/fisher_L{L}.h5)")
    p.add_argument("--out", default=os.path.join(ROOT, "img_results", "first_exploration.png"))
    return p.parse_args()


def get_data(args, temps):
    path = args.data or os.path.join(ROOT, "data", f"fisher_L{args.L}.h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    want = args.chains * args.samples
    out = {}
    for k, T in enumerate(temps):
        d = None
        try:
            d = dataset.load_samples(path, args.L, T, 0.0)
            if len(d["E_J"]) != want or np.unique(d["chain_id"]).size != args.chains:
                d = None
        except (KeyError, OSError):
            d = None
        if d is None:
            seed = int(np.random.SeedSequence([args.seed, k, args.L]).generate_state(1)[0])
            t0 = time.time()
            S, E, M, cid = mc.run_chains(args.L, T, 0.0, 1.0, args.chains, args.burn,
                                         args.samples, args.thin, seed)
            dataset.save_samples(path, args.L, T, 0.0, S, E, M, cid, overwrite=True)
            d = dataset.load_samples(path, args.L, T, 0.0)
            print(f"  MC  T={T:.4f}: generated {want} configs in {time.time() - t0:.1f}s")
        else:
            print(f"  MC  T={T:.4f}: loaded {want} configs from {os.path.relpath(path, ROOT)}")
        out[T] = d
    return out


def physical_FTT(E_J, chain_id, T):
    """F_TT^phys = Var(E_J)/T^4 at h=0, with a jackknife-over-chains error bar."""
    est = np.var(E_J) / T ** 4
    chains = np.unique(chain_id)
    jk = np.array([np.var(E_J[chain_id != c]) / T ** 4 for c in chains])
    n = chains.size
    err = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    return float(est), float(err)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    temps = [float(t) for t in np.round(np.linspace(args.Tmin, args.Tmax, args.nT), 6)]
    L = args.L
    t_start = time.time()
    print(f"L={L}, T grid={temps}, T_c={T_C:.4f}, rcond={args.rcond}, clf_rcond={args.clf_rcond}")

    data = get_data(args, temps)
    n_train_chains = args.chains // 2
    tr, ev = {}, {}
    for T in temps:
        d = data[T]
        m = d["chain_id"] < n_train_chains
        tr[T] = {k: v[m] for k, v in d.items() if isinstance(v, np.ndarray)}
        ev[T] = {k: v[~m] for k, v in d.items() if isinstance(v, np.ndarray)}

    def pool(part):
        X = np.concatenate([part[T]["spins"] for T in temps]).astype(np.float32)
        Ts = np.concatenate([np.full(len(part[T]["E_J"]), T, np.float32) for T in temps])
        return X, Ts

    Xtr, Ttr = pool(tr)
    Xev, Tev = pool(ev)
    ytr, yev = (Ttr >= T_C).astype(np.float32), (Tev >= T_C).astype(np.float32)

    # ---------------- classifiers ----------------
    clfs = {"SimpleMLP": SimpleMLP(L), "SimpleCNN": SimpleCNN(L)}
    rf = {name: {"r_f": [], "r_f_heldout": [], "rank": []} for name in clfs}
    for name, model in clfs.items():
        t0 = time.time()
        torch.manual_seed(args.seed)
        train_classifier(model, Xtr, ytr, epochs=args.clf_epochs, batch_size=256, lr=2e-3,
                         seed=args.seed)
        vl, va = evaluate_classifier(model, Xev, yev)
        P = sum(p.numel() for p in model.parameters())
        print(f"  {name}: P={P}, eval loss={vl:.4f}, eval acc={va:.4f} ({time.time() - t0:.1f}s)")
        for T in temps:
            e = ev[T]
            # Split across sub-blocks of chains so both folds sample all Z2 sectors at low T
            block_groups = np.arange(len(e["E_J"])) // 250
            res = fisher.compute_classifier_rf(model, e["spins"], e["E_J"], T, h=0.0,
                                               rcond=args.clf_rcond, groups=block_groups,
                                               seed=args.seed)
            for k in rf[name]:
                rf[name][k].append(res[k])
            print(f"    T={T:.3f}  r_f={res['r_f']:.4f}  r_f(held-out)={res['r_f_heldout']:.4f}"
                  f"  rank={res['rank']}/{res['n_params']}  N={res['n']}")

    # ---------------- generative model ----------------
    t0 = time.time()
    torch.manual_seed(args.seed)
    gen = SimpleGenCNN(z2_equivariant=True)
    hist = train_generator(gen, Xtr, Ttr, 0.0, epochs=args.gen_epochs, batch_size=256, lr=2e-3,
                           seed=args.seed)
    print(f"  SimpleGenCNN(z2): P_active={gen.n_effective_parameters()}, "
          f"train NLL={hist['train_nll'][-1]:.4f} ({time.time() - t0:.1f}s)")

    spin_cols = ~fisher.conditioning_columns(gen)
    gres = {k: [] for k in (
        "r_T", "r_T_heldout", "S_T", "S_T_heldout", "r_h", "rank",
        "r_T_nocond", "r_T_nocond_heldout", "S_T_nocond", "S_T_nocond_heldout", "rank_nocond",
        "r_T_spinW", "r_T_spinW_heldout", "S_T_spinW", "S_T_spinW_heldout", "rank_spinW",
        "r_T_L1", "r_T_L1_heldout", "S_T_L1", "S_T_L1_heldout", "rank_L1",
        "F_TT_model", "F_TT_model_err", "F_Th_model", "F_TT_phys", "F_TT_phys_err", "val_nll_per_site"
    )}

    for k, T in enumerate(temps):
        t1 = time.time()
        e = ev[T]
        gres["val_nll_per_site"].append(evaluate_generator(gen, e["spins"].astype(np.float32), T, 0.0) / L ** 2)
        torch.manual_seed(args.seed + 1 + k)
        xq = gen.sample(args.n_model_samples, L, T, 0.0)
        sc = fisher.compute_generative_scores(gen, xq, T, 0.0)

        bias_cols = torch.tensor(["bias" in lbl for lbl in sc.param_labels], dtype=torch.bool)
        spinW_cols = spin_cols & (~bias_cols)
        l1_cols = spin_cols & torch.tensor(["conv1.weight" in lbl for lbl in sc.param_labels], dtype=torch.bool)

        sch = fisher.compute_schur_invariant(sc.u_theta, torch.stack([sc.u_T, sc.u_h], 1),
                                             rcond=args.rcond, seed=args.seed)
        snc = fisher.compute_schur_invariant(sc.u_theta[:, spin_cols], sc.u_T,
                                             rcond=args.rcond, seed=args.seed)
        ssw = fisher.compute_schur_invariant(sc.u_theta[:, spinW_cols], sc.u_T,
                                             rcond=args.rcond, seed=args.seed)
        sl1 = fisher.compute_schur_invariant(sc.u_theta[:, l1_cols], sc.u_T,
                                             rcond=args.rcond, seed=args.seed)

        for prefix, d_sub in (("nocond", snc), ("spinW", ssw), ("L1", sl1)):
            gres[f"r_T_{prefix}"].append(d_sub["r_c"])
            gres[f"r_T_{prefix}_heldout"].append(d_sub["r_c_heldout"])
            gres[f"S_T_{prefix}"].append(d_sub["S_c"])
            gres[f"S_T_{prefix}_heldout"].append(d_sub["S_c_heldout"])
            gres[f"rank_{prefix}"].append(d_sub["rank"])

        cf = fisher.model_control_fisher(sc.u_T, sc.u_h)
        uT2 = (sc.u_T ** 2).numpy()
        Fp, Fp_err = physical_FTT(data[T]["E_J"], data[T]["chain_id"], T)
        gres["r_T"].append(float(sch["r_c"][0]))
        gres["r_T_heldout"].append(float(sch["r_c_heldout"][0]))
        gres["S_T"].append(float(sch["S_c"][0]))
        gres["S_T_heldout"].append(float(sch["S_c_heldout"][0]))
        gres["r_h"].append(float(sch["r_c"][1]))
        gres["rank"].append(sch["rank"])
        gres["F_TT_model"].append(cf["F_TT_model"])
        gres["F_TT_model_err"].append(float(uT2.std() / np.sqrt(uT2.size)))
        gres["F_Th_model"].append(cf["F_Th_model"])
        gres["F_TT_phys"].append(Fp)
        gres["F_TT_phys_err"].append(Fp_err)
        print(f"    T={T:.3f}  r_T(full)={gres['r_T_heldout'][-1]:.6f}"
              f"  r_T(spinW)={ssw['r_c_heldout']:.4f}  r_T(L1)={sl1['r_c_heldout']:.4f}"
              f"  F_TT model/phys={cf['F_TT_model']:.4g}/{Fp:.4g}  ({time.time() - t1:.1f}s)")

    # ---------------- figure ----------------
    Ta = np.array(temps)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.5))

    a = ax[0, 0]
    for name, c in (("SimpleMLP", "tab:blue"), ("SimpleCNN", "tab:orange")):
        a.plot(Ta, rf[name]["r_f_heldout"], "o-", color=c, label=f"{name} (held-out)")
        a.plot(Ta, rf[name]["r_f"], "s--", color=c, alpha=0.5, mfc="none", label=f"{name} (in-sample)")
    a.set_ylim(0.0, 1.05)
    a.set_ylabel(r"$r_f(T)$")
    a.set_title(r"(1) Classifier: $R^2$ of $s_T$ on $\nabla_\theta f$")

    a = ax[0, 1]
    a.semilogy(Ta, np.clip(1.0 - np.array(gres["r_T_heldout"]), 1e-8, None), "o-", color="tab:green", label=r"full $\theta$ (held-out)")
    a.semilogy(Ta, np.clip(1.0 - np.array(gres["r_T_nocond_heldout"]), 1e-8, None), "s--", color="tab:brown", label=r"$\theta\setminus W_{T,h}$ (held-out)")
    a.semilogy(Ta, np.clip(1.0 - np.array(gres["r_T_spinW_heldout"]), 1e-8, None), "^-", color="tab:blue", label=r"spin weights $\theta\setminus(W_{T,h}\cup b)$")
    a.semilogy(Ta, np.clip(1.0 - np.array(gres["r_T_L1_heldout"]), 1e-8, None), "d-", color="tab:red", label=r"Layer-1 spin weights (conv1)")
    a.set_ylabel(r"$1 - r_T(T)$")
    a.set_title(r"(2a) SimpleGenCNN($\mathbb{Z}_2$): Schur deficit $1 - r_T$")

    a = ax[1, 0]
    a.semilogy(Ta, gres["F_TT_model"], "^-", color="k", label=r"$F_{TT}^{model}$")
    a.semilogy(Ta, np.clip(gres["S_T_heldout"], 1e-12, None), "o-", color="tab:green", label=r"$S_T$ full $\theta$")
    a.semilogy(Ta, np.clip(gres["S_T_nocond_heldout"], 1e-12, None), "s--", color="tab:brown",
               label=r"$S_T$ ($\theta\setminus W_{T,h}$)")
    a.semilogy(Ta, np.clip(gres["S_T_spinW_heldout"], 1e-12, None), "^-", color="tab:blue",
               label=r"$S_T$ (spin weights, no bias)")
    a.semilogy(Ta, np.clip(gres["S_T_L1_heldout"], 1e-12, None), "d-", color="tab:red",
               label=r"$S_T$ (Layer-1 spin weights)")
    a.set_ylabel("information")
    a.set_title(r"(2b) Schur residual $S_T = F_{TT}^{model}(1-r_T)$")

    a = ax[1, 1]
    a.errorbar(Ta, gres["F_TT_phys"], yerr=gres["F_TT_phys_err"], fmt="o-", color="tab:purple",
               capsize=3, label=r"$F_{TT}^{phys}$ = Var$(E_J)/T^4$ (MC)")
    a.errorbar(Ta, gres["F_TT_model"], yerr=gres["F_TT_model_err"], fmt="^--", color="k",
               capsize=3, label=r"$F_{TT}^{model} = E_q[u_T^2]$")
    a.set_ylabel(r"$F_{TT}$")
    a.set_title("(3) Model fidelity")

    for a in ax.ravel():
        a.axvline(T_C, color="gray", ls=":", lw=1)
        a.set_xlabel("T")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f"Stage 1 first exploration: L={L}, h=0, {args.chains}x{args.samples} MC configs/T, "
                 f"{args.n_model_samples} model samples/T, rcond={args.rcond:g}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)

    summary = {"args": vars(args), "T": temps, "T_c": T_C, "classifier": rf, "generative": gres,
               "gen_train_nll": hist["train_nll"]}
    js = os.path.splitext(args.out)[0] + ".json"
    with open(js, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"saved {os.path.relpath(args.out, ROOT)} and {os.path.relpath(js, ROOT)} "
          f"(total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
