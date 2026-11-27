#!/usr/bin/env python3
"""Main-benchmark charts of the report, regenerated from the five-seed study on the
345-image test split (~/btp2027/seed_study_345/results, 3 methods x 4 N x 5 seeds).
Same style as gen_figures_v3.py. Calibration as in the report: CC-SAT-DA tau = 0.35,
the other methods uncalibrated. Writes PNGs, a LaTeX table and numbers.json to
~/btp2027/report_figs/v4/."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
S = os.environ.get("CCSATDA_STUDY_DIR", os.path.join(REPO, "results", "seed_study_345"))
OUT = os.path.join(REPO, "figures", "main"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from run_study import metrics_from_logits, LOG_RATIO  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#D9DDE3", "grid.linestyle": "--", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})
SUP, SAT, CC = "#E0A526", "#6C8EBF", "#0B2545"
ACC2 = "#8DA9C4"
NAMES = {"sup": "Target-only supervised", "satda": "SAT-DA", "cc": "CC-SAT-DA (ours)"}
COL = {NAMES["sup"]: SUP, NAMES["satda"]: SAT, NAMES["cc"]: CC}
CL = ["BCC", "MEL", "NV", "AKIEC", "BKL"]
NK = ["50", "100", "200", "full"]; NS = ["50", "100", "200", "1,608"]
x = np.arange(4)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("saved", name)


def bar_labels(ax, bars, vals, fmt, dy=0.0, size=8, tops=None):
    for i, (b, v) in enumerate(zip(bars, vals)):
        top = v if tops is None else tops[i]
        ax.text(b.get_x() + b.get_width() / 2, top + dy, fmt.format(v), ha="center", va="bottom", fontsize=size,
                color="#333333")


def predict(logits, tau):
    return (logits + tau * LOG_RATIO).argmax(1)


# ------------------------------------------------------------------ load all runs
R = {}
for f in glob.glob(f"{S}/results/*.json"):
    r = json.load(open(f))
    tau = 0.35 if r["method"] == "cc" else 0.0
    tl, ty = np.array(r["test_logits"]), np.array(r["test_labels"])
    m = metrics_from_logits(tl, ty, tau=tau)
    pred = predict(tl, tau)
    p, rc, f1, _ = precision_recall_fscore_support(ty, pred, labels=range(5), zero_division=0)
    m.update(prec=p, rec=rc, f1c=f1, cm=confusion_matrix(ty, pred, labels=range(5)), seed=r["seed"])
    if r["method"] == "cc":
        m0 = metrics_from_logits(tl, ty, tau=0.0)
        m["raw"] = m0
    R.setdefault((r["method"], r["n"]), []).append(m)
assert all(len(v) == 5 for v in R.values()) and len(R) == 12, {k: len(v) for k, v in R.items()}


def arr(meth, key, sc=1.0):
    return np.array([[run[key] * sc for run in R[(meth, n)]] for n in NK])  # 4 x 5


M = {}
for k, name in NAMES.items():
    M[name] = dict(acc=arr(k, "accuracy", 100), f1=arr(k, "f1_macro"), auc=arr(k, "auroc_macro_ovr"),
                   mel=arr(k, "mel_found"), bcc=np.array([[run["rec"][0] * 100 for run in R[(k, n)]] for n in NK]))

# ------------------------------------------------------------------ numbers for the text
num = {}
for k, name in NAMES.items():
    for i, n in enumerate(NK):
        runs = R[(k, n)]
        d = {key: [float(M[name][key][i].mean()), float(M[name][key][i].std())] for key in ("acc", "f1", "auc", "mel", "bcc")}
        mp = sum(r["mel_pred"] for r in runs); mf = sum(r["mel_found"] for r in runs)
        d["mel_pooled"] = [mf, mp, 40]
        d["mel_runs"] = [r["mel_found"] for r in sorted(runs, key=lambda r: r["seed"])]
        d["f1_runs"] = [r["f1_macro"] for r in sorted(runs, key=lambda r: r["seed"])]
        d["any_mel_runs"] = int(sum(r["mel_found"] > 0 for r in runs))
        cm = sum(r["cm"] for r in runs)
        rec = np.diag(cm) / cm.sum(1); prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
        d["pooled_rec"] = rec.tolist(); d["pooled_prec"] = prec.tolist()
        d["pooled_f1"] = (2 * rec * prec / np.maximum(rec + prec, 1e-9)).tolist()
        d["cm"] = cm.tolist()
        num[f"{k}_{n}"] = d
# wins of cc over others per run (same seed)
for other in ("sup", "satda"):
    for n in NK:
        a = {r["seed"]: r["f1_macro"] for r in R[("cc", n)]}; b = {r["seed"]: r["f1_macro"] for r in R[(other, n)]}
        num[f"cc_beats_{other}_f1_{n}"] = int(sum(a[s] > b[s] for s in a))
        a = {r["seed"]: r["accuracy"] for r in R[("cc", n)]}; b = {r["seed"]: r["accuracy"] for r in R[(other, n)]}
        num[f"cc_beats_{other}_acc_{n}"] = int(sum(a[s] > b[s] for s in a))
# calibration (cc), all N
for n in NK:
    runs = R[("cc", n)]
    def agg(sel):
        acc = np.mean([sel(r)["accuracy"] for r in runs]) * 100
        f1 = np.mean([sel(r)["f1_macro"] for r in runs]); auc = np.mean([sel(r)["auroc_macro_ovr"] for r in runs])
        mf = sum(sel(r)["mel_found"] for r in runs); mp = sum(sel(r)["mel_pred"] for r in runs)
        return dict(acc=acc, f1=f1, auc=auc, mel_found=mf, mel_pred=mp, mel_rec=mf / 40 * 100, mel_prec=mf / max(mp, 1) * 100)
    num[f"calib_{n}"] = {"tau0": agg(lambda r: r["raw"]), "tau035": agg(lambda r: r)}
json.dump(num, open(f"{OUT}/numbers.json", "w"), indent=1)

# ------------------------------------------------------------------ LaTeX table
rows = []
for i, n in enumerate(NK):
    block = []
    best = {key: max(M[nm][key][i].mean() for nm in NAMES.values()) for key in ("acc", "f1", "auc", "mel")}
    for j, (k, nm) in enumerate(NAMES.items()):
        cells = []
        for key, f, sc in (("acc", "{:.1f}", 1), ("f1", "{:.3f}", 1), ("auc", "{:.3f}", 1), ("mel", "{:.1f}", 1)):
            mu, sd = M[nm][key][i].mean(), M[nm][key][i].std()
            c = f"\\pmstd{{{f.format(mu)}}}{{{f.format(sd)}}}"
            if f.format(mu) == f.format(best[key]):
                c = f"\\best{{{c}}}"
            cells.append(c)
        label = "\\textbf{CC-SAT-DA (ours)}" if k == "cc" else nm
        first = f"\\multirow{{3}}{{*}}{{{NS[i]}}}" if j == 0 else ""
        block.append(f"{first} & {label} & " + " & ".join(cells) + " \\\\")
    rows.append("\n".join(block))
tex = ("\\begin{tabular}{@{}l l c c c c@{}}\n\\toprule\n$N$ & \\textbf{Method} & \\textbf{Accuracy (\\%)} & "
       "\\textbf{Macro F1} & \\textbf{AUROC} & \\makecell{\\textbf{MEL found}\\\\ \\textbf{(of 8)}} \\\\\n\\midrule\n"
       + "\n\\midrule\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(f"{OUT}/main_benchmark.tex", "w").write(tex)

# ------------------------------------------------------------------ 3. main results: grouped bars (mean, error bar = std)
w = 0.26
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
spec = [("acc", "Accuracy (%)", "{:.1f}", (40, 85), 0.6), ("f1", "Macro F1", "{:.3f}", (0.3, 0.84), 0.006),
        ("auc", "Macro AUROC", "{:.3f}", (0.6, 1.0), 0.004), ("mel", "Melanomas found (of 8)", "{:.1f}", (0, 8.5), 0.1)]
for ax, (k, title, fmt, ylim, dy) in zip(axes.ravel(), spec):
    for i, (name, d) in enumerate(M.items()):
        mu, sd = d[k].mean(1), d[k].std(1)
        bars = ax.bar(x + (i - 1) * w, mu, w, color=COL[name], label=name, edgecolor="white", linewidth=0.6)
        ax.errorbar(x + (i - 1) * w, mu, yerr=sd, fmt="none", ecolor="#444444", elinewidth=0.9, capsize=2.5)
        bar_labels(ax, bars, mu, fmt, dy, size=7.2, tops=mu + sd)
    ax.set_xticks(x); ax.set_xticklabels(NS); ax.set_xlabel("Labelled target images $N$")
    ax.set_title(title); ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig_main_bars")

# ------------------------------------------------------------------ 4. learning curves (mean +- std band)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, k, title, sc in [(axes[0], "f1", "Macro F1", 1), (axes[1], "mel", "Melanoma recall (%)", 12.5)]:
    for name, d in M.items():
        mu, sd = d[k].mean(1) * sc, d[k].std(1) * sc
        ax.plot(x, mu, marker="o", ms=7, lw=2.4, color=COL[name], label=name, mec="white", mew=1.2)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.12, color=COL[name], lw=0)
        if name == NAMES["cc"]:
            others = np.max([dd[k].mean(1) * sc for nn, dd in M.items() if nn != name], axis=0)
            for xi, yi, oi in zip(x, mu, others):
                below = k == "mel" and yi < oi  # another curve sits above: put the label under the point
                ax.annotate(f"{yi:.3f}" if k == "f1" else f"{yi:.1f}", (xi, yi), textcoords="offset points",
                            xytext=(9, -4) if below else (0, 9), ha="left" if below else "center", fontsize=8.5, color=CC, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(NS); ax.set_xlabel("Labelled target images $N$"); ax.set_title(title)
axes[0].set_ylim(0.3, 0.82); axes[1].set_ylim(-5, 100)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "fig_curves")

# ------------------------------------------------------------------ 5. radar (means)
labels = ["Accuracy", "Macro F1", "AUROC", "Melanoma\nrecall", "BCC\nrecall"]
ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), subplot_kw=dict(polar=True))
for ax, idx, title in [(axes[0], 1, "$N = 100$"), (axes[1], 3, "$N = 1{,}608$")]:
    for name, d in M.items():
        v = [d["acc"][idx].mean() / 100, d["f1"][idx].mean(), d["auc"][idx].mean(), d["mel"][idx].mean() / 8,
             d["bcc"][idx].mean() / 100]
        v += v[:1]
        ax.plot(ang, v, color=COL[name], lw=2.2, label=name)
        ax.fill(ang, v, color=COL[name], alpha=0.13)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=8, color="#777777")
    ax.set_title(title, pad=16); ax.grid(color="#D9DDE3")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig_radar")

# ------------------------------------------------------------------ 6. gains (mean of paired per-seed differences)
cc, sup, sat = M[NAMES["cc"]], M[NAMES["sup"]], M[NAMES["satda"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
for ax, k, t, fmt, ylim in [(axes[0], "f1", "Gain in macro F1", "{:+.3f}", (-0.04, 0.2)),
                            (axes[1], "acc", "Gain in accuracy (points)", "{:+.1f}", (-6, 12.5))]:
    d1 = cc[k].mean(1) - sup[k].mean(1); d2 = cc[k].mean(1) - sat[k].mean(1)
    for off, d, col, lab in [(-0.19, d1, SUP, "over target-only supervised"), (0.19, d2, SAT, "over SAT-DA")]:
        bars = ax.bar(x + off, d, 0.38, color=col, label=lab, edgecolor="white")
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] * 0.015 if v >= 0 else -ylim[1] * 0.08),
                    fmt.format(v), ha="center", fontsize=8.5)
    ax.axhline(0, color="#333333", lw=0.9); ax.set_xticks(x); ax.set_xticklabels(NS); ax.set_ylim(*ylim)
    ax.set_xlabel("Labelled target images $N$"); ax.set_title(t); ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "fig_gains")

# ------------------------------------------------------------------ 7. melanomas found: runs (dots) + mean (bar)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for i, (name, d) in enumerate(M.items()):
    xs = x + (i - 1) * 0.25
    mu = d["mel"].mean(1)
    ax.bar(xs, mu, 0.22, color=COL[name], alpha=0.25)
    for xi, vals in zip(xs, d["mel"]):
        ax.scatter(xi + rng.uniform(-0.06, 0.06, len(vals)), vals, s=34, color=COL[name], edgecolors="white",
                   linewidths=0.7, zorder=3)
    ax.scatter([], [], s=60, color=COL[name], label=name)
    for xi, v in zip(xs, mu):
        ax.text(xi, max(v, d["mel"][list(xs).index(xi)].max()) + 0.3, f"{v:.1f}", ha="center", fontsize=8.5,
                color=COL[name], fontweight="bold")
ax.axhline(8, color="#C0392B", ls=":", lw=1.2); ax.text(-0.45, 8.12, "8 test melanomas", ha="left", fontsize=9, color="#C0392B")
ax.set_xticks(x); ax.set_xticklabels(NS); ax.set_ylim(-0.3, 9.0); ax.set_yticks(range(0, 9))
ax.set_xlabel("Labelled target images $N$"); ax.set_ylabel("Melanomas found"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.13))
save(fig, "fig_mel_found")

# ------------------------------------------------------------------ 8. confusion matrices (full N, pooled over 5 seeds)
cmap = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", "#C9D6E8", "#0B2545"])
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
for ax, k in zip(axes, ("sup", "cc")):
    cm = np.array(num[f"{k}_full"]["cm"]); title = NAMES[k]
    rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{rn[i, j]*100:.0f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if rn[i, j] > 0.55 else "#222222", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(CL); ax.set_yticks(range(5)); ax.set_yticklabels(CL)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title, color=COL[title]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
save(fig, "fig_confusion")

# ------------------------------------------------------------------ 9. per-class F1 heatmap + P/R bars (full N, mean over seeds)
pc = {nm: {"prec": np.array(num[f"{k}_full"]["pooled_prec"]), "rec": np.array(num[f"{k}_full"]["pooled_rec"]), "f1c": np.array(num[f"{k}_full"]["pooled_f1"])} for k, nm in NAMES.items()}  # pooled over 5 seeds, as the confusion matrices
f1m = np.array([pc[nm]["f1c"] for nm in NAMES.values()])
fig, ax = plt.subplots(figsize=(8, 2.7))
ax.imshow(f1m, cmap=cmap, vmin=0.5, vmax=0.9, aspect="auto")
for i in range(3):
    for j in range(5):
        best = f1m[i, j] == f1m[:, j].max()
        ax.text(j, i, f"{f1m[i, j]:.3f}", ha="center", va="center", fontsize=10,
                color="white" if f1m[i, j] > 0.74 else "#222222", fontweight="bold" if best else "normal")
ax.set_xticks(range(5)); ax.set_xticklabels(CL); ax.set_yticks(range(3)); ax.set_yticklabels(list(NAMES.values())); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Per-class F1 score (full target set, 5 seeds pooled)")
save(fig, "fig_f1_heatmap")
num["perclass_full"] = {nm: {key: v.tolist() for key, v in d.items()} for nm, d in pc.items()}
json.dump(num, open(f"{OUT}/numbers.json", "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, key, title in [(axes[0], "rec", "Recall (%)"), (axes[1], "prec", "Precision (%)")]:
    for i, nm in enumerate(NAMES.values()):
        vals = pc[nm][key] * 100
        bars = ax.bar(np.arange(5) + (i - 1) * 0.27, vals, 0.27, color=COL[nm], label=nm, edgecolor="white")
        bar_labels(ax, bars, vals, "{:.0f}", 0.8, size=7.5)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(CL); ax.set_title(title); ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "fig_perclass_bars")

# ------------------------------------------------------------------ 10. calibration (full-N CC-SAT-DA, mean of 5 seeds)
c0, c1 = num["calib_full"]["tau0"], num["calib_full"]["tau035"]
names = ["Accuracy", "Macro F1", "AUROC", "MEL recall", "MEL precision"]
v0 = [c0["acc"], c0["f1"] * 100, c0["auc"] * 100, c0["mel_rec"], c0["mel_prec"]]
v1 = [c1["acc"], c1["f1"] * 100, c1["auc"] * 100, c1["mel_rec"], c1["mel_prec"]]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
b0 = ax.bar(np.arange(5) - 0.19, v0, 0.38, color=ACC2, label=r"without calibration ($\tau=0$)")
b1 = ax.bar(np.arange(5) + 0.19, v1, 0.38, color=CC, label=r"with calibration ($\tau=0.35$)")
bar_labels(ax, b0, v0, "{:.1f}", 0.8, 8.5); bar_labels(ax, b1, v1, "{:.1f}", 0.8, 8.5)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(names); ax.set_ylabel("Value (%)"); ax.set_ylim(40, 103)
ax.grid(axis="x", visible=False); ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.13))
save(fig, "fig_calibration")
print(json.dumps({k: v for k, v in num.items() if k.startswith(("cc_beats", "calib"))}, indent=1))
