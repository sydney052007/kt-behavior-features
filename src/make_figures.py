"""三張報告草稿圖（matplotlib）。只從現有 csv 讀數字，不重算、不重跑實驗。

    python3 src/make_figures.py
輸出到 figures/（每張圖 PNG 300 dpi + SVG），圖背後的數據表在 figures/data/*.csv。

  fig2_ladder              資訊階梯：LR／RF × R0→R0d→R1 的 A ROC-AUC（各 fold 與平均）與每個 fold 的增量
                           來源：05_report_tables/g_fold_auc.csv、a_main_A.csv、b_paired_A.csv
  fig3_behavior_groups     行為特徵群的邊際貢獻（LR）：加進 R0d、從 R1 拿掉
                           來源：02_analysis/ablation_lr_paired.csv
  fig4_cross_school        學生分組 vs 學校群分組的 R0d−R0、R1−R0d（LR、RF）
                           來源：05_report_tables/e_cross_school.csv

視覺規則（三張圖一致）：顏色 = 特徵組（R0／R0d／R1，Okabe-Ito 色盲友善配色）；形狀 = 模型（LR 圓、RF 方）；
fold 的點是各自的 fold 值，粗橫線／粗線是 csv 裡的平均（不是在這裡重算的）。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

R2, R5 = "results/02_analysis/", "results/05_report_tables/"
FIG, DATA = "figures/", "figures/data/"
FOLDS = [f"fold{k}" for k in range(5)]
COLORS = {"R0": "#0072B2", "R0d": "#E69F00", "R1": "#009E73"}     # Okabe-Ito：藍、橘、綠
MARKERS = {"LR": "o", "RF": "s"}
GRAY = "#8c8c8c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "svg.hashsalt": "kt-behavior-features", "svg.fonttype": "path"})


def rd(path):
    """float_precision='round_trip'：讀回的浮點數與檔案裡的字串完全對應。"""
    return pd.read_csv(path, float_precision="round_trip", dtype={"max_depth": str}, keep_default_na=False, na_values=[""])


BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=1.0)       # 標注的白底（不透明），避免被線穿過


def fmt(x, signed=False, nd=3):
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


# ====================================================================== 數據表（圖背後）
def data_fig2():
    g, a, b = rd(R5 + "g_fold_auc.csv"), rd(R5 + "a_main_A.csv"), rd(R5 + "b_paired_A.csv")
    levels = []
    for _, r in g.iterrows():
        mean = a[(a["model_type"] == r["model_type"]) & (a["features"] == r["features"])]["roc_auc_mean"].iloc[0]
        for k in range(5):
            levels.append(dict(model_type=r["model_type"], features=r["features"], fold=k, roc_auc=r[f"roc_fold{k}"], mean_roc_auc=mean))
    inc = []
    for mt, an in [("LR", "main"), ("RF", "own_best")]:
        for step, comp in [("R0->R0d", "R0d - R0"), ("R0d->R1", "R1 - R0d")]:
            r = b[(b["model_type"] == mt) & (b["analysis"] == an) & (b["comparison"] == comp)].iloc[0]
            for k in range(5):
                inc.append(dict(model_type=mt, step=step, fold=k, delta_auc=r[f"fold{k}"], mean_delta_auc=r["delta_mean"]))
    return pd.DataFrame(levels), pd.DataFrame(inc)


GROUPS = ["hint", "time", "att", "scaffold"]
GAP3 = 1.4                       # 圖 3 各組的 x 間距（留空間給平均值標注）


def data_fig3():
    p = rd(R2 + "ablation_lr_paired.csv")
    p = p[p["eval"] == "A"]
    rows = []
    for panel, prefix, ref in [("a_add_to_R0d", "R0d+", "R0d"), ("b_remove_from_R1", "R1-", "R1")]:
        for g in GROUPS:
            r = p[(p["model"] == prefix + g) & (p["reference"] == ref)]
            assert len(r) == 1, (prefix, g)
            r = r.iloc[0]
            for k in range(5):
                rows.append(dict(panel=panel, group=g, reference=ref, fold=k, delta_auc=r[f"fold{k}"], mean_delta_auc=r["delta_auc"]))
    return pd.DataFrame(rows)


def data_fig3_refs():
    """圖 3 的兩條水平參考線：R1 − R0d 的 A 平均（LR main，來自 b_paired_A）；(b) 取負號（拿掉全部行為特徵 = R0d vs R1）。"""
    b = rd(R5 + "b_paired_A.csv")
    v = b[(b["model_type"] == "LR") & (b["analysis"] == "main") & (b["comparison"] == "R1 - R0d")]["delta_mean"].iloc[0]
    return pd.DataFrame([
        dict(panel="a_add_to_R0d", label="all behavior groups (R1 − R0d)", value=v, source="05_report_tables/b_paired_A.csv: LR main R1 - R0d delta_mean"),
        dict(panel="b_remove_from_R1", label="all behavior groups removed (R0d vs R1)", value=-v, source="same value with the sign flipped"),
    ])


def data_fig4():
    e = rd(R5 + "e_cross_school.csv")
    rows = []
    for _, r in e.iterrows():
        panel = "a_R0d_minus_R0" if r["comparison"] == "R0d - R0" else "b_R1_minus_R0d"
        for k in range(5):
            rows.append(dict(panel=panel, model_type=r["model_type"], split=r["split"], fold=k, delta_auc=r[f"fold{k}"],
                             mean_delta_auc=r["delta_mean"], is_largest_group_fold=bool(r["split"] == "school_group" and k == 0)))
    return pd.DataFrame(rows)


# ====================================================================== 繪圖
def panel_label(ax, text):
    ax.text(-0.02, 1.08, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right", gid="outside")


def fig2(levels, inc):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), sharey="row")
    feats = ["R0", "R0d", "R1"]
    for j, mt in enumerate(["LR", "RF"]):
        ax = axes[0, j]
        d = levels[levels["model_type"] == mt]
        for k in range(5):                                                   # 每個 fold 一條細灰線
            ax.plot(range(3), [d[(d["features"] == f) & (d["fold"] == k)]["roc_auc"].iloc[0] for f in feats],
                    color=GRAY, lw=0.9, alpha=0.8, zorder=1)
        means = [d[d["features"] == f]["mean_roc_auc"].iloc[0] for f in feats]
        ax.plot(range(3), means, color="black", lw=2.8, zorder=2)             # 平均：粗線 + 標記
        for x, f, m in zip(range(3), feats, means):
            ax.plot(x, m, marker=MARKERS[mt], ms=10, color=COLORS[f], mec="black", mew=1.2, zorder=3)
            # R0 的標注放左上（粗線往右上走，放正上方會被線擋住）；其餘放正上方
            ax.annotate(fmt(m), (x, m), textcoords="offset points", xytext=(-9, 7) if f == "R0" else ((0, 22) if f == "R0d" else (0, 12)),
                        ha="right" if f == "R0" else "center", fontsize=9, fontweight="bold", bbox=BOX, zorder=4)
        ax.set_xticks(range(3), feats)
        ax.set_xlim(-0.4, 2.4)
        ax.set_title(mt, fontsize=12)
        ax.text(0.0, -0.17, "y-axis truncated", transform=ax.transAxes, fontsize=7.5, style="italic", color="#444444",
                ha="left", va="top", gid="outside")
        if j == 0:
            ax.set_ylabel("ROC-AUC (setting A, 5-fold)")
        ax = axes[1, j]
        d = inc[inc["model_type"] == mt]
        for x, (step, col) in enumerate([("R0->R0d", COLORS["R0d"]), ("R0d->R1", COLORS["R1"])]):
            s = d[d["step"] == step].sort_values("fold")
            ax.scatter(x + np.linspace(-0.14, 0.14, 5), s["delta_auc"], marker=MARKERS[mt], s=48, color=col, edgecolor="black", linewidth=0.8, zorder=3)
            m = s["mean_delta_auc"].iloc[0]
            ax.hlines(m, x - 0.27, x + 0.27, color="black", lw=2.4, zorder=2)
            ax.annotate(fmt(m, True), (x + 0.27, m), textcoords="offset points", xytext=(5, 0), va="center", fontsize=9, fontweight="bold", bbox=BOX)
        ax.axhline(0, color="black", lw=1, ls="--", zorder=1)
        ax.set_xticks([0, 1], ["R0 → R0d", "R0d → R1"])
        ax.set_xlim(-0.6, 1.75)
        if j == 0:
            ax.set_ylabel("Paired ΔAUC per fold (same y-axis for both steps)")
    panel_label(axes[0, 0], "(a)")
    panel_label(axes[1, 0], "(b)")
    handles = [Line2D([], [], color=GRAY, lw=1), Line2D([], [], color="black", lw=2.8),
               Line2D([], [], marker="o", color="none", mfc=COLORS["R0"], mec="black", ms=8, label="R0"),
               Line2D([], [], marker="o", color="none", mfc=COLORS["R0d"], mec="black", ms=8),
               Line2D([], [], marker="o", color="none", mfc=COLORS["R1"], mec="black", ms=8),
               Line2D([], [], marker="o", color="none", mfc="white", mec="black", ms=8),
               Line2D([], [], marker="s", color="none", mfc="white", mec="black", ms=8)]
    labels = ["Single fold", "Mean of 5 folds", "R0", "R0d", "R1", "LR (circle)", "RF (square)"]
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Information ladder: AUC per feature set and the gain of each step", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def strip(ax, x, values, mean, color, marker, hollow=False, fold0_marker=None, width=0.27, jitter=0.12, nd=3, label_dy=0):
    """一組的 fold 點 + 平均橫線。fold0_marker：把 fold 0 換成別的標記（用在學校群分組的最大組）。"""
    xs = x + np.linspace(-jitter, jitter, len(values))
    for k, (xx, v) in enumerate(zip(xs, values)):
        mk = fold0_marker if (fold0_marker and k == 0) else marker
        ax.scatter(xx, v, marker=mk, s=52, facecolor="white" if hollow else color, edgecolor=color if hollow else "black",
                   linewidth=1.6 if hollow else 0.8, zorder=3)
    ax.hlines(mean, x - width, x + width, color="black", lw=2.4, zorder=2)
    # 標註放在點群右側；接近零線時上下錯開（label_dy），避免壓到零線
    ax.annotate(fmt(mean, True, nd), (x + width, mean), textcoords="offset points", xytext=(6, label_dy), va="center", fontsize=8.5,
                fontweight="bold", bbox=BOX, zorder=4, gid="mean-label")


def fig3(d, refs):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), sharey=True)
    lim = 1.4 * max(d["delta_auc"].abs().max(), refs["value"].abs().max())              # 兩個子圖共用 y 範圍（只是呈現）
    xmax = (len(GROUPS) - 1) * GAP3 + 1.4                                              # 右側留空間給最後一組的標註
    for ax, panel, ref, title in [(axes[0], "a_add_to_R0d", "R0d", "Add the group to R0d"), (axes[1], "b_remove_from_R1", "R1", "Remove the group from R1")]:
        for i, g in enumerate(GROUPS):
            s = d[(d["panel"] == panel) & (d["group"] == g)].sort_values("fold")
            mean = s["mean_delta_auc"].iloc[0]
            dy = (10 if mean >= 0 else -10) if abs(mean) < 0.0008 else 0                 # 接近零線的標註上下錯開
            strip(ax, i * GAP3, s["delta_auc"].to_numpy(), mean, COLORS[ref], MARKERS["LR"], nd=4, label_dy=dy)
        ax.axhline(0, color="black", lw=1, ls="--", zorder=1)
        r = refs[refs["panel"] == panel].iloc[0]                                          # 全部行為特徵一起加入／拿掉
        ax.axhline(r["value"], color="#444444", lw=1.3, ls=(0, (6, 2, 1, 2)), zorder=1)
        up = r["value"] > 0
        ax.annotate(r["label"], (xmax - 0.05, r["value"]), textcoords="offset points", xytext=(0, 5 if up else -5), ha="right",
                    va="bottom" if up else "top", fontsize=8, color="#333333", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
                    gid="refline-label")
        ax.set_xticks([i * GAP3 for i in range(len(GROUPS))], GROUPS)
        ax.set_xlim(-0.6, xmax)
        ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Behavior feature group")
    axes[0].set_ylabel("ΔAUC vs. reference (LR only, setting A)")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    handles = [Line2D([], [], marker="o", color="none", mfc=COLORS["R0d"], mec="black", ms=8),
               Line2D([], [], marker="o", color="none", mfc=COLORS["R1"], mec="black", ms=8),
               Line2D([], [], color="black", lw=2.4), Line2D([], [], color="#444444", lw=1.3, ls=(0, (6, 2, 1, 2)))]
    fig.legend(handles, ["Single fold; reference = R0d", "Single fold; reference = R1", "Mean of 5 folds", "All groups together (±(R1 − R0d))"],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Marginal contribution of behavior feature groups (LR only)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def fig4(d):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    # 每個子圖四個位置：LR（學生分組、學校群分組）、RF（學生分組、學校群分組）；間距留給平均值標注
    slots = {("LR", "student_split"): 0.0, ("LR", "school_group"): 1.95, ("RF", "student_split"): 4.4, ("RF", "school_group"): 6.35}
    specs = [(axes[0], "a_R0d_minus_R0", "R0d", "R0d − R0", "ΔAUC (R0d − R0)", 3), (axes[1], "b_R1_minus_R0d", "R1", "R1 − R0d", "ΔAUC (R1 − R0d)", 4)]
    for ax, panel, col, title, ylabel, nd in specs:
        for (mt, split), x in slots.items():
            s = d[(d["panel"] == panel) & (d["model_type"] == mt) & (d["split"] == split)].sort_values("fold")
            if s.empty:                                                   # 這一組沒有跑（RF 的學校群 R0d−R0）
                ax.text(x, 0.5, "RF R0 not run\non school split", ha="center", va="center", fontsize=8.5, style="italic",
                        color="#555555", transform=ax.get_xaxis_transform())
                continue
            school = split == "school_group"
            strip(ax, x, s["delta_auc"].to_numpy(), s["mean_delta_auc"].iloc[0], COLORS[col], MARKERS[mt], hollow=school,
                  fold0_marker="D" if school else None, nd=nd)
        ax.axhline(0, color="black", lw=1, ls="--", zorder=1)
        ax.set_xticks(list(slots.values()), ["Student\nsplit", "School\ngroup"] * 2, fontsize=9)
        ax.set_xlim(-0.7, 8.0)
        for mt, xm in [("LR", 0.975), ("RF", 5.375)]:
            ax.text(xm, -0.21, mt, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11, fontweight="bold", gid="outside")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.002), hi)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    handles = [Line2D([], [], marker="o", color="none", mfc=COLORS["R0d"], mec="black", ms=8),
               Line2D([], [], marker="o", color="none", mfc=COLORS["R1"], mec="black", ms=8),
               Line2D([], [], marker="o", color="none", mfc="#bbbbbb", mec="black", ms=8),
               Line2D([], [], marker="o", color="none", mfc="white", mec="#444444", mew=1.6, ms=8),
               Line2D([], [], marker="D", color="none", mfc="white", mec="#444444", mew=1.6, ms=7),
               Line2D([], [], color="black", lw=2.4)]
    labels = ["Panel (a): R0d − R0", "Panel (b): R1 − R0d", "Student split (filled marker)", "School-group split (hollow marker)",
              "School-group fold 0 = largest group (diamond)", "Mean of 5 folds"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.115, "Note: panels (a) and (b) use different y-axis scales.   LR = circle, RF = square.", ha="center", va="center",
             fontsize=8, style="italic", color="#444444")
    fig.suptitle("Student split vs. school-group split", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.17, 1, 0.96))
    return fig


def _bbox(t, r):
    from matplotlib.transforms import Bbox
    bb = t.get_window_extent(r)
    p = t.get_bbox_patch()
    return Bbox.union([bb, p.get_window_extent(r)]) if p is not None else bb


def check_layout(fig):
    """標註的版面檢查：不互相覆蓋、不超出圖框／座標軸、不壓到資料點與水平線（零線、參考線、平均線）。回傳問題清單。"""
    from matplotlib.collections import LineCollection, PathCollection
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    probs, items = [], []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text():
                items.append((ax, t, _bbox(t, r)))
    fb = fig.bbox
    for ax, t, bb in items:
        name = t.get_text().replace("\n", " ")
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
            probs.append(f"超出圖框：{name}")
        if t.get_gid() != "outside":
            ab = ax.get_window_extent(r)
            if bb.x0 < ab.x0 - 1 or bb.x1 > ab.x1 + 1 or bb.y0 < ab.y0 - 1 or bb.y1 > ab.y1 + 1:
                probs.append(f"超出座標軸：{name}")
    boxes = [(t.get_text().replace("\n", " "), bb) for _, t, bb in items]
    for ax in fig.axes:                                                       # 刻度文字也算（只算真的畫出來的：刻度位置在顯示範圍內）
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = min(lim), max(lim)
            for loc, tl in zip(axis.get_majorticklocs(), axis.get_ticklabels()):
                if tl.get_text() and lo - 1e-9 <= loc <= hi + 1e-9:
                    boxes.append(("tick:" + tl.get_text(), tl.get_window_extent(r)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                probs.append(f"標註互相覆蓋：{boxes[i][0]} ↔ {boxes[j][0]}")
    for ax, t, bb in items:
        name = t.get_text().replace("\n", " ")
        pts = []
        for c in ax.collections:
            if isinstance(c, PathCollection):
                pts += list(ax.transData.transform(c.get_offsets()))
        for ln in ax.lines:
            if ln.get_marker() not in ("None", None, "") and len(ln.get_xdata()):
                pts += list(ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()])))
        for px, py in pts:
            if bb.x0 - 4 <= px <= bb.x1 + 4 and bb.y0 - 4 <= py <= bb.y1 + 4:
                probs.append(f"標註壓到資料點：{name}")
                break
        segs = []
        for c in ax.collections:
            if isinstance(c, LineCollection):
                segs += [ax.transData.transform(sg) for sg in c.get_segments()]
        for ln in ax.lines:
            y = np.asarray(ln.get_ydata(), float)
            if len(y) >= 2 and np.all(y == y[0]) and ln.get_marker() in ("None", None, ""):
                segs.append(ln.get_transform().transform(np.column_stack([ln.get_xdata(), y])))
        for sg in segs:
            if sg[:, 1].min() >= bb.y0 and sg[:, 1].max() <= bb.y1 and sg[:, 0].min() < bb.x1 and sg[:, 0].max() > bb.x0:
                probs.append(f"標註壓到水平線：{name}")
                break
    return probs


def save(fig, name):
    fig.savefig(FIG + name + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG + name + ".svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    levels, inc = data_fig2()
    d3, d4, refs3 = data_fig3(), data_fig4(), data_fig3_refs()
    levels.to_csv(DATA + "fig2_ladder_levels.csv", index=False)
    inc.to_csv(DATA + "fig2_ladder_increments.csv", index=False)
    d3.to_csv(DATA + "fig3_behavior_groups.csv", index=False)
    d4.to_csv(DATA + "fig4_cross_school.csv", index=False)
    refs3.to_csv(DATA + "fig3_reference_lines.csv", index=False)
    for name, fig in [("fig2_ladder", fig2(levels, inc)), ("fig3_behavior_groups", fig3(d3, refs3)), ("fig4_cross_school", fig4(d4))]:
        probs = check_layout(fig)
        print(f"[layout] {name}：" + ("沒有標註互相覆蓋、超出圖框或壓到點／線" if not probs else "; ".join(probs)))
        assert not probs, probs
        save(fig, name)
    print("已存", sorted(os.listdir(FIG)), sorted(os.listdir(DATA)))


if __name__ == "__main__":
    main()
