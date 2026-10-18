"""Draw the HW1 plots from results.json into figures/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

METRICS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]
LABELS = {"recall@10": "Recall@10", "mrr@10": "MRR@10", "ndcg@10": "NDCG@10",
          "hit@10": "Hit@10", "precision@10": "Precision@10"}
K_LIST = [1, 5, 10, 100, 500]

with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def line(ax, xs, ys, color, label=None):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, label=label)


def plot_knn():
    xs = list(range(len(K_LIST)))
    fig, grid = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = grid.flatten()
    for ax, m in zip(axes, METRICS):
        item = [R[f"ItemKNN_k{k}"]["test"][m] for k in K_LIST]
        user = [R[f"UserKNN_k{k}"]["test"][m] for k in K_LIST]
        line(ax, xs, item, BLUE, "ItemKNN")
        line(ax, xs, user, ORANGE, "UserKNN")
        style(ax, f"{LABELS[m]} vs k", "k (number of neighbors)", LABELS[m])
        ax.set_xticks(xs, [str(k) for k in K_LIST])
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    axes[5].legend(handles, labels, loc="center", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "knn_k.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bpr_single(keys, xticks, title, xlabel, fname):
    xs = list(range(len(keys)))
    ys = [R[k]["test"]["ndcg@10"] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    line(ax, xs, ys, BLUE)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK)
    style(ax, title, xlabel, "NDCG@10")
    ax.set_xticks(xs, xticks)
    ax.margins(y=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    plt.close(fig)


def plot_bpr():
    embs = [32, 64, 128, 256]
    regs = [0.01, 0.1, 0.5]
    plot_bpr_single([f"BPR_emb{e}_reg0.1" for e in embs], [str(e) for e in embs],
                    "BPR: NDCG@10 vs Embedding Size (reg_lambda=0.1)",
                    "Embedding Size", "bpr_emb.png")
    plot_bpr_single([f"BPR_emb64_reg{r}" for r in regs], [str(r) for r in regs],
                    "BPR: NDCG@10 vs Regularization (embedding_size=64)",
                    "Regularization Lambda", "bpr_reg.png")


def plot_item2item():
    models = ["ItemKNN_default", "SLIMElastic", "EASE"]
    names = ["ItemKNN (k=100)", "SLIM", "EASE"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.26
    for i, m in enumerate(METRICS):
        for j, (model, color) in enumerate(zip(models, ["#a3a29c", BLUE, ORANGE])):
            v = R[model]["test"][m]
            ax.bar(i + (j - 1) * width, v, width - 0.02, color=color,
                   label=names[j] if i == 0 else None)
    ax.set_xticks(range(len(METRICS)), [LABELS[m] for m in METRICS])
    style(ax, "ItemKNN vs SLIM vs EASE (test, ml-100k)", "", "Score")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "item2item.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_knn()
    plot_item2item()
    if all(f"BPR_emb{e}_reg0.1" in R for e in [32, 64, 128, 256]) and \
            all(f"BPR_emb64_reg{r}" in R for r in [0.01, 0.1, 0.5]):
        plot_bpr()
    print("saved to", FIG_DIR)
