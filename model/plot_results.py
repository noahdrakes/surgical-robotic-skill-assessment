import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _fmt_metric(
    estimate: float,
    spread_type: str | None,
    spread_value: float,
    lower: float,
    upper: float,
) -> str:
    if pd.isna(estimate):
        return "NA"
    if spread_type == "std" and not pd.isna(spread_value):
        return f"{estimate:.4f} ± {spread_value:.4f}"
    if pd.isna(lower) or pd.isna(upper):
        return f"{estimate:.4f}"
    return f"{estimate:.4f} [{lower:.4f}, {upper:.4f}]"


def _get_row(
    df: pd.DataFrame,
    feature_set: str,
    metric_name: str,
    class_name: str,
    eval_level: str,
) -> pd.Series | None:
    rows = df[
        (df["feature_set"] == feature_set)
        & (df["metric_name"] == metric_name)
        & (df["class_name"] == class_name)
        & (df["eval_level"] == eval_level)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _feature_set_label(feature_set: str) -> str:
    mapping = {
        "all_no_forcen": "All (No Force Norm.)",
        "all": "All Features",
        "kinematics": "Kinematics",
        "kinetics": "Kinetics",
        "force": "Force",
    }
    return mapping.get(feature_set, feature_set.replace("_", " ").title())


def build_paper_table(metrics_df: pd.DataFrame, eval_level: str = "pooled") -> pd.DataFrame:
    metrics_df = metrics_df.copy()
    metrics_df["estimate"] = pd.to_numeric(metrics_df["estimate"], errors="coerce")
    metrics_df["spread_value"] = pd.to_numeric(metrics_df["spread_value"], errors="coerce")
    metrics_df["lower"] = pd.to_numeric(metrics_df["lower"], errors="coerce")
    metrics_df["upper"] = pd.to_numeric(metrics_df["upper"], errors="coerce")

    feature_sets = sorted(metrics_df["feature_set"].dropna().unique())
    class_names = [
        c
        for c in metrics_df["class_name"].dropna().unique()
        if c.lower() != "overall"
    ]

    table_rows = []
    for feature_set in feature_sets:
        row = {
            "Feature Set": _feature_set_label(feature_set),
        }

        for auc_metric, col_name in [
            ("auc_macro", "AUC Macro"),
            ("auc_weighted", "AUC Weighted"),
        ]:
            metric_row = _get_row(metrics_df, feature_set, auc_metric, "overall", eval_level)
            if metric_row is None:
                row[col_name] = "NA"
            else:
                row[col_name] = _fmt_metric(
                    metric_row["estimate"],
                    metric_row.get("spread_type"),
                    metric_row.get("spread_value"),
                    metric_row["lower"],
                    metric_row["upper"],
                )

        for class_name in class_names:
            sens_row = _get_row(metrics_df, feature_set, "sensitivity", class_name, eval_level)
            spec_row = _get_row(metrics_df, feature_set, "specificity", class_name, eval_level)

            sens_col = f"{class_name} Sens."
            spec_col = f"{class_name} Spec."

            row[sens_col] = (
                _fmt_metric(
                    sens_row["estimate"],
                    sens_row.get("spread_type"),
                    sens_row.get("spread_value"),
                    sens_row["lower"],
                    sens_row["upper"],
                )
                if sens_row is not None
                else "NA"
            )
            row[spec_col] = (
                _fmt_metric(
                    spec_row["estimate"],
                    spec_row.get("spread_type"),
                    spec_row.get("spread_value"),
                    spec_row["lower"],
                    spec_row["upper"],
                )
                if spec_row is not None
                else "NA"
            )

        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    return table_df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.fillna("").values.tolist()]

    def _escape(cell: str) -> str:
        return cell.replace("|", "\\|")

    header_line = "| " + " | ".join(_escape(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = [
        "| " + " | ".join(_escape(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def dataframe_to_latex_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    latex_df = df.copy()
    latex_df = latex_df.replace({r"_": r"\_"}, regex=True)
    body = latex_df.to_latex(
        index=False,
        escape=False,
        na_rep="NA",
        column_format="l" * len(df.columns),
        caption=caption,
        label=label,
        bold_rows=False,
    )
    return body


def save_table_png(df: pd.DataFrame, out_png: str, dpi: int = 300) -> None:
    def _plot_col_label(col: str) -> str:
        if col in {"Feature Set", "AUC Macro", "AUC Weighted"}:
            return col
        return f"{col}\n(estimate ± std or CI)"

    def _col_width_factor(col: str) -> float:
        if col in {"AUC Macro", "AUC Weighted"}:
            return 0.75
        return 1.0

    rows, cols = df.shape
    base_col_width = 1.0 / cols if cols else 1.0
    col_widths = [base_col_width * _col_width_factor(c) for c in df.columns]

    fig_w = max(12, cols * 1.8)
    fig_h = max(2.5, rows * 0.65 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=[_plot_col_label(c) for c in df.columns],
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#efefef")
        cell.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a paper-ready statistical metrics table from louo_metrics_table.csv."
    )
    parser.add_argument(
        "--metrics_csv",
        default="model/results/louo_metrics_table.csv",
        help="Path to long-format louo metrics table CSV.",
    )
    parser.add_argument(
        "--out_csv",
        default="model/results/louo_paper_table.csv",
        help="Path to save paper-ready table CSV.",
    )
    parser.add_argument(
        "--out_md",
        default="model/results/louo_paper_table.md",
        help="Path to save paper-ready table Markdown.",
    )
    parser.add_argument(
        "--out_tex",
        default="model/results/louo_paper_table.tex",
        help="Path to save LaTeX table (booktabs).",
    )
    parser.add_argument(
        "--caption",
        default=(
            "LOUO pooled performance across feature sets. Values are point estimates with "
            "95% CI when available."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:louo_performance",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--out_png",
        default="model/results/louo_paper_table.png",
        help="Path to save paper-ready table as PNG image.",
    )
    parser.add_argument(
        "--eval_level",
        default="pooled",
        help="Evaluation level to render from metrics CSV (e.g., pooled, seed_sweep).",
    )
    parser.add_argument(
        "--png_dpi",
        type=int,
        default=300,
        help="DPI for PNG output.",
    )
    args = parser.parse_args()

    metrics_df = pd.read_csv(args.metrics_csv)
    available_levels = set(metrics_df["eval_level"].dropna().unique().tolist())
    if args.eval_level in available_levels:
        metrics_df = metrics_df[metrics_df["eval_level"] == args.eval_level].copy()
    elif available_levels:
        fallback = sorted(available_levels)[0]
        print(
            f"Requested eval_level '{args.eval_level}' not found. "
            f"Falling back to '{fallback}'."
        )
        metrics_df = metrics_df[metrics_df["eval_level"] == fallback].copy()
    table_df = build_paper_table(metrics_df, eval_level=args.eval_level)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    table_df.to_csv(args.out_csv, index=False)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(dataframe_to_markdown(table_df))
            f.write("\n")
    if args.out_tex:
        os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
        with open(args.out_tex, "w", encoding="utf-8") as f:
            f.write(dataframe_to_latex_booktabs(table_df, args.caption, args.label))
            f.write("\n")
    if args.out_png:
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        save_table_png(table_df, args.out_png, dpi=args.png_dpi)

    print(f"Saved paper table CSV: {args.out_csv}")
    if args.out_md:
        print(f"Saved paper table Markdown: {args.out_md}")
    if args.out_tex:
        print(f"Saved paper table LaTeX: {args.out_tex}")
    if args.out_png:
        print(f"Saved paper table PNG: {args.out_png}")


if __name__ == "__main__":
    main()
