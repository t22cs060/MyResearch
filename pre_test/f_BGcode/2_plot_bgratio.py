import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 設定 ===
DIR_PATH = "pre_test/f_BGcode/figure/"
CSV_PATH = f"pre_test/f_BGcode/label_ratios.csv"
OUTPUT_IMG_PATH = f"{DIR_PATH}abel_distribution.png"
LABELS = ["Background", "Objective", "Methods", "Results", "Conclusions"]


# === 各ラベルについて平均を計算 
def overoll_plot():
    avg_ratios = {}
    for label in LABELS:
        abstract_col = f"abstract_{label.lower()}_ratio"
        summary_col = f"summary_{label.lower()}_ratio"
        avg_ratio = (df[abstract_col].mean() + df[summary_col].mean()) / 2
        avg_ratios[label] = avg_ratio

    # --- グラフ描画 
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(avg_ratios.keys()), y=list(avg_ratios.values()), palette="mako")
    plt.title("Average Label Ratio Across Abstracts and Summaries")
    plt.ylabel("Ratio")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # --- 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH)
    plt.close()

    print(f"✅ グラフ画像を保存しました: {OUTPUT_IMG_PATH}")


# === 各ラベルについて平均を計算（abstとsummを分けて）
def plot_label_distribution(df, section="abstract", output_path="output.png"):
    ratios = {}
    for label in LABELS:
        col = f"{section}_{label.lower()}_ratio"
        ratios[label] = df[col].mean()

    # --- グラフ描画
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(ratios.keys()), y=list(ratios.values()), palette="Set2")
    plt.title(f"Average Label Ratio in {section.capitalize()}")
    plt.ylabel("Ratio")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ {section.capitalize()} グラフ保存完了: {output_path}")


# === 各ラベルごとに，データの分布を出力（abstとsummをひとつに）
def plot_distribution_by_label(df, label):
    abstract_col = f"abstract_{label.lower()}_ratio"
    summary_col = f"summary_{label.lower()}_ratio"

    # データを整形（long形式）
    plot_df = pd.DataFrame({
        "Ratio": pd.concat([df[abstract_col], df[summary_col]], ignore_index=True),
        "Section": ["Abstract"] * len(df) + ["Summary"] * len(df)
    })

    # 色設定: abstract = lightblue, summary = eLife blue
    color_map = {"Abstract": "lightblue", "Summary": "#2ca02c"}

    plt.figure(figsize=(8, 5))
    sns.histplot(data=plot_df, x="Ratio", hue="Section", kde=True, bins=20, palette=color_map, alpha=0.7)
    plt.title(f"Distribution of '{label}' Ratio in Abstract vs Summary")
    plt.xlabel("Ratio")
    plt.ylabel("Document Count")
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = f"{DIR_PATH}label_distribution_{label}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✅ {label} 分布グラフ保存: {out_path}")


# === 実行 ===
df = pd.read_csv(CSV_PATH) # CSV読み込み 
overoll_plot()
plot_label_distribution(df, section="abstract", output_path=f"{DIR_PATH}abstract_label_distribution.png")
plot_label_distribution(df, section="summary", output_path=f"{DIR_PATH}summary_label_distribution.png")

for label in LABELS: 
    plot_distribution_by_label(df, label)
