"""Generate LaTeX table fragments from pipeline results.

Reads saved CSV/TXT results from results/tables/ and writes .tex table
fragments to paper/tables/. Each fragment is a standalone table environment
that can be included in the paper via LaTeX input commands.
"""

import logging
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "tables"
PAPER_TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"

# Readable names for uncertainty categories
CAT_LABELS = {
    "epistemic_hedge": "Epistemic Hedge",
    "evidential_marker": "Evidential Marker",
    "explicit_uncertainty": "Explicit Uncertainty",
    "probability_language": "Probability Language",
    "modal_hedge": "Modal Hedge",
    "approximator": "Approximator",
}


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(char, f"\\{char}")
    return s


def _fmt_int(n) -> str:
    """Format integer with thousands separator for LaTeX."""
    return f"{int(n):,}".replace(",", "{,}")


def _fmt_pct(val, decimals=1) -> str:
    """Format a proportion as a percentage string."""
    return f"{float(val) * 100:.{decimals}f}"


def _fmt_p(p) -> str:
    """Format a p-value for LaTeX."""
    p = float(p)
    if p < 0.001:
        exp = f"{p:.1e}"
        base, power = exp.split("e")
        return f"$< 0.001$"
    elif p < 0.01:
        return f"${p:.3f}$"
    elif p < 0.05:
        return f"${p:.3f}$"
    else:
        return f"${p:.3f}$"


def _sig_stars(p) -> str:
    """Return significance stars for a p-value."""
    p = float(p)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _write_table(path: Path, content: str) -> None:
    """Write a LaTeX table fragment to file."""
    path.write_text(content)
    logger.info("Wrote %s", path.name)


# ─── Table Generators ──────────────────────────────────────────────────


def generate_dataset_overview():
    """Table 1: Dataset overview and model family breakdown."""
    eda_path = RESULTS_DIR / "eda_summary.csv"
    family_path = RESULTS_DIR / "family_reasoning_availability.csv"
    if not eda_path.exists() or not family_path.exists():
        logger.warning("Skipping dataset overview: source files missing")
        return

    eda = pl.read_csv(eda_path)
    fam = pl.read_csv(family_path)

    # Extract key metrics from eda_summary
    metrics = {row["metric"]: row["value"] for row in eda.iter_rows(named=True)}

    total = int(metrics["total_rows"])
    with_reasoning = int(metrics["rows_with_reasoning"])
    with_response = int(metrics["rows_with_response"])
    with_both = int(metrics["rows_with_both"])
    n_models = int(metrics["n_unique_models"])
    reasoning_pct = float(metrics["reasoning_pct"])
    median_len = int(float(metrics["reasoning_len_median"]))
    p95_len = int(float(metrics["reasoning_len_p95"]))

    # Build table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset overview. The \textsc{real-slop} corpus contains multi-model interactions; only rows with non-null reasoning traces are used for analysis.}",
        r"\label{tab:dataset}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
        f"Total interactions & {_fmt_int(total)} \\\\",
        f"With reasoning traces & {_fmt_int(with_reasoning)} ({reasoning_pct:.1f}\\%) \\\\",
        f"With response & {_fmt_int(with_response)} \\\\",
        f"With both reasoning \\& response & {_fmt_int(with_both)} \\\\",
        f"Unique models & {n_models} \\\\",
        f"Reasoning length (median) & {_fmt_int(median_len)} chars \\\\",
        f"Reasoning length (95th pct) & {_fmt_int(p95_len)} chars \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Reasoning availability by model family}} \\",
        r"\midrule",
        r"Family & With reasoning (\%) \\",
        r"\midrule",
    ]

    # Sort by reasoning availability descending
    fam_sorted = fam.sort("pct_reasoning", descending=True)
    for row in fam_sorted.iter_rows(named=True):
        family = _escape_latex(row["model_family"])
        pct = float(row["pct_reasoning"])
        n = int(row["with_reasoning"])
        lines.append(f"{family} & {_fmt_int(n)} ({pct:.1f}\\%) \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_table(PAPER_TABLES_DIR / "tab_dataset_overview.tex", "\n".join(lines))


def generate_lexicon_validation():
    """Table 2: Lexicon validation (agreement + category rates)."""
    agree_path = RESULTS_DIR / "classifier_agreement.csv"
    rates_path = RESULTS_DIR / "lexicon_category_rates.csv"
    if not agree_path.exists() or not rates_path.exists():
        logger.warning("Skipping lexicon validation: source files missing")
        return

    agree = pl.read_csv(agree_path)
    rates = pl.read_csv(rates_path)

    agree_dict = {row["metric"]: row["value"] for row in agree.iter_rows(named=True)}
    kappa = float(agree_dict["cohens_kappa"])
    llm_rate = float(agree_dict["llm_positive_rate"])
    lex_rate = float(agree_dict["lexical_positive_rate"])
    n_samples = int(float(agree_dict["n_samples"]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Lexicon validation against LLM classifier (Qwen 2.5 7B). Cohen's $\kappa$ indicates substantial agreement. Per-category detection rates are from the lexicon applied to 500 sampled reasoning traces.}",
        r"\label{tab:validation}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\multicolumn{2}{l}{\textit{Inter-method agreement} ($n$ = " + _fmt_int(n_samples) + r")} \\",
        r"\midrule",
        f"Cohen's $\\kappa$ & {kappa:.3f} \\\\",
        f"LLM positive rate & {llm_rate * 100:.1f}\\% \\\\",
        f"Lexical positive rate & {lex_rate * 100:.1f}\\% \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Lexicon category detection rates}} \\",
        r"\midrule",
        r"Category & \% of sentences \\",
        r"\midrule",
    ]

    for row in rates.iter_rows(named=True):
        cat = CAT_LABELS.get(row["category"], row["category"])
        pct = float(row["pct_of_sentences"])
        n = int(row["n_sentences_with"])
        lines.append(f"{cat} & {pct:.2f}\\% ($n$={n}) \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_table(PAPER_TABLES_DIR / "tab_lexicon_validation.tex", "\n".join(lines))


def generate_positional_deciles():
    """Table 3: Uncertainty by position decile (main result)."""
    path = RESULTS_DIR / "uncertainty_by_decile.csv"
    if not path.exists():
        logger.warning("Skipping positional deciles: %s not found", path)
        return

    df = pl.read_csv(path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Uncertainty expression rate by position decile in reasoning chains. Rates represent the proportion of sentences containing at least one uncertainty marker. The overall rate peaks in deciles 2--4 and declines toward the conclusion.}",
        r"\label{tab:deciles}",
        r"\small",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Decile & $N$ & Overall & Epist. & Evid. & Prob. & Modal & Approx. \\",
        r"\midrule",
    ]

    # Find peak decile for bolding
    rates = df["uncertainty_rate"].to_list()
    peak_idx = rates.index(max(rates))

    for i, row in enumerate(df.iter_rows(named=True)):
        d = int(row["position_decile"])
        n = _fmt_int(row["n_sentences"])
        overall = _fmt_pct(row["uncertainty_rate"])
        epist = _fmt_pct(row["cat_epistemic_hedge_rate"])
        evid = _fmt_pct(row["cat_evidential_marker_rate"], 2)
        prob = _fmt_pct(row["cat_probability_language_rate"])
        modal = _fmt_pct(row["cat_modal_hedge_rate"])
        approx = _fmt_pct(row["cat_approximator_rate"], 2)

        if i == peak_idx:
            lines.append(f"\\textbf{{{d}}} & {n} & \\textbf{{{overall}}} & {epist} & {evid} & {prob} & {modal} & {approx} \\\\")
        else:
            lines.append(f"{d} & {n} & {overall} & {epist} & {evid} & {prob} & {modal} & {approx} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_table(PAPER_TABLES_DIR / "tab_positional_deciles.tex", "\n".join(lines))


def generate_confidence_filtering():
    """Table 5: Confidence filtering paired tests."""
    path = RESULTS_DIR / "filtering_paired_tests.csv"
    if not path.exists():
        logger.warning("Skipping confidence filtering: %s not found", path)
        return

    df = pl.read_csv(path)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Confidence filtering: paired comparison of uncertainty rates in reasoning traces versus final responses (Wilcoxon signed-rank test, one-sided). Significance: {*}$p < .05$, {**}$p < .01$, {***}$p < .001$.}",
        r"\label{tab:filtering}",
        r"\small",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Category & Reasoning (\%) & Response (\%) & $\Delta$ (\%) & Sig. \\",
        r"\midrule",
    ]

    for row in df.iter_rows(named=True):
        test = row["test"]
        if test == "overall":
            label = "\\textbf{Overall}"
        else:
            label = CAT_LABELS.get(test, test)
        r_mean = float(row["reasoning_mean"]) * 100
        resp_mean = float(row["response_mean"]) * 100
        delta = r_mean - resp_mean
        p = float(row["p_value"])
        stars = _sig_stars(p)

        lines.append(f"{label} & {r_mean:.2f} & {resp_mean:.2f} & {delta:+.2f} & {stars} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_table(PAPER_TABLES_DIR / "tab_confidence_filtering.tex", "\n".join(lines))


def generate_model_families():
    """Table 4: Model family comparison summary."""
    path = RESULTS_DIR / "uncertainty_by_decile_model.csv"
    if not path.exists():
        logger.warning("Skipping model families: %s not found", path)
        return

    df = pl.read_csv(path)

    # Compute summary per family
    families = df["model_family"].unique().sort().to_list()
    summaries = []
    for fam in families:
        subset = df.filter(pl.col("model_family") == fam)
        total_n = int(subset["n_sentences"].sum())
        if total_n < 100:
            continue  # skip families with very few sentences
        mean_rate = float(subset["uncertainty_rate"].mean())
        peak_row = subset.sort("uncertainty_rate", descending=True).row(0, named=True)
        peak_decile = int(peak_row["position_decile"])
        peak_rate = float(peak_row["uncertainty_rate"])
        summaries.append({
            "family": fam,
            "n_sentences": total_n,
            "mean_rate": mean_rate,
            "peak_decile": peak_decile,
            "peak_rate": peak_rate,
        })

    # Sort by mean rate descending
    summaries.sort(key=lambda x: x["mean_rate"], reverse=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Uncertainty rates by model family. Mean rate is averaged across all position deciles. Families with fewer than 100 total sentences are excluded.}",
        r"\label{tab:families}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model Family & $N$ Sentences & Mean Rate (\%) & Peak Decile & Peak Rate (\%) \\",
        r"\midrule",
    ]

    for s in summaries:
        fam = _escape_latex(s["family"])
        lines.append(
            f"{fam} & {_fmt_int(s['n_sentences'])} & {s['mean_rate'] * 100:.1f} & "
            f"{s['peak_decile']} & {s['peak_rate'] * 100:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_table(PAPER_TABLES_DIR / "tab_model_families.tex", "\n".join(lines))


def _parse_mixedlm_summary(path: Path) -> tuple[dict, list[dict]]:
    """Parse a statsmodels MixedLM summary text file.

    Returns:
        Tuple of (header_info dict, list of coefficient row dicts).
    """
    text = path.read_text()
    lines = text.strip().split("\n")

    header = {}
    coefs = []
    separator_count = 0

    for line in lines:
        # Extract header info
        if "No. Observations:" in line:
            match = re.search(r"No\. Observations:\s+(\d+)", line)
            if match:
                header["n_obs"] = int(match.group(1))
        if "No. Groups:" in line:
            match = re.search(r"No\. Groups:\s+(\d+)", line)
            if match:
                header["n_groups"] = int(match.group(1))
        if "Converged:" in line:
            header["converged"] = "Yes" in line
        if "Log-Likelihood:" in line:
            match = re.search(r"Log-Likelihood:\s+([-\d.]+)", line)
            if match:
                header["log_likelihood"] = float(match.group(1))

        # Track separator lines
        if re.match(r"^-+$", line.strip()):
            separator_count += 1
            continue

        # Coefficient rows appear after the second separator
        if separator_count >= 2 and line.strip() and not re.match(r"^=+$", line.strip()):
            parts = line.split()
            if len(parts) >= 5:
                # Handle multi-word predictor names like C(model_family)[T.xxx]
                # Find where the numeric values start
                numeric_start = None
                for j, p in enumerate(parts):
                    try:
                        float(p.replace("-", ""))
                        numeric_start = j
                        break
                    except ValueError:
                        continue

                if numeric_start is not None and numeric_start > 0:
                    name = " ".join(parts[:numeric_start])
                    vals = parts[numeric_start:]
                    if len(vals) >= 4:
                        coefs.append({
                            "predictor": name,
                            "coef": vals[0],
                            "stderr": vals[1],
                            "z": vals[2],
                            "p": vals[3],
                            "ci_lo": vals[4] if len(vals) > 4 else "",
                            "ci_hi": vals[5] if len(vals) > 5 else "",
                        })

    return header, coefs


def generate_mixed_effects():
    """Table 6: Mixed-effects regression results."""
    pos_path = RESULTS_DIR / "model_positional_summary.txt"
    filt_path = RESULTS_DIR / "model_filtering_summary.txt"

    if not pos_path.exists():
        logger.warning("Skipping mixed effects: positional summary not found")
        return

    pos_header, pos_coefs = _parse_mixedlm_summary(pos_path)

    # Key predictors to include (skip model family dummies)
    key_predictors = {"Intercept", "normalized_position", "position_sq",
                      "log_reasoning_len", "nsfw_int", "Group Var"}

    # Readable labels
    pred_labels = {
        "Intercept": "Intercept",
        "normalized_position": "Position (linear)",
        "position_sq": "Position$^2$ (quadratic)",
        "log_reasoning_len": "Log reasoning length",
        "nsfw_int": "NSFW content",
        "Group Var": "Random intercept var.",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mixed-effects linear model results. Panel (a): positional uncertainty model (DV: uncertainty presence per sentence, " + _fmt_int(pos_header.get("n_obs", 0)) + r" observations across " + _fmt_int(pos_header.get("n_groups", 0)) + r" interactions). Panel (b): filtering ratio model. Model family dummies were included in (a) but are omitted for brevity (all $p > .05$).}",
        r"\label{tab:regression}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Predictor & Coef. & Std.Err. & $z$ & $p$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{(a) Positional uncertainty model}} \\",
        r"\midrule",
    ]

    for c in pos_coefs:
        if c["predictor"] in key_predictors:
            label = pred_labels.get(c["predictor"], c["predictor"])
            p_val = float(c["p"]) if c["p"] else 1.0
            stars = _sig_stars(p_val)
            if c["predictor"] == "Group Var":
                lines.append(f"{label} & {c['coef']} & {c['stderr']} & -- & -- \\\\")
            else:
                lines.append(f"{label} & {c['coef']} & {c['stderr']} & {c['z']} & {_fmt_p(p_val)}{stars} \\\\")

    # Filtering model
    if filt_path.exists():
        filt_header, filt_coefs = _parse_mixedlm_summary(filt_path)
        lines.extend([
            r"\midrule",
            r"\multicolumn{5}{l}{\textit{(b) Filtering ratio model} ($n$ = " + _fmt_int(filt_header.get("n_obs", 0)) + r")} \\",
            r"\midrule",
        ])

        filt_labels = {
            "Intercept": "Intercept",
            "log_reasoning_len": "Log reasoning length",
            "Group Var": "Random intercept var.",
        }

        for c in filt_coefs:
            label = filt_labels.get(c["predictor"], c["predictor"])
            p_val = float(c["p"]) if c["p"] else 1.0
            stars = _sig_stars(p_val)
            if c["predictor"] == "Group Var":
                lines.append(f"{label} & {c['coef']} & -- & -- & -- \\\\")
            else:
                lines.append(f"{label} & {c['coef']} & {c['stderr']} & {c['z']} & {_fmt_p(p_val)}{stars} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    _write_table(PAPER_TABLES_DIR / "tab_mixed_effects.tex", "\n".join(lines))


def main():
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    generate_dataset_overview()
    generate_lexicon_validation()
    generate_positional_deciles()
    generate_confidence_filtering()
    generate_model_families()
    generate_mixed_effects()

    logger.info("\nAll LaTeX tables generated in %s", PAPER_TABLES_DIR)


if __name__ == "__main__":
    main()
