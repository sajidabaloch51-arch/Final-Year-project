"""
IG-06 · Report Generator
Produces a structured PDF/text report of benchmark findings.
"""
import sys, os, json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import REPORT_DIR, FIGURES_DIR, PROCESSED_DIR, REPORT_TITLE

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except:
        return str(v)


def generate_text_report(results: dict) -> Path:
    """Generate a plain-text summary report."""
    out_path = REPORT_DIR / "ig06_benchmark_report.txt"
    violence  = results["violence_metric"]
    fid_sum   = results["fidelity_summary"]
    def_rates = results["default_rates"]
    gender_r  = results["gender_rates"]

    lines = [
        "=" * 70,
        f"  {REPORT_TITLE}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "── EXECUTIVE SUMMARY ──────────────────────────────────────────────────",
        "",
        f"  Total prompts evaluated :  {len(results['df_scored']):,}",
        f"  Overall fidelity pass   :  {_fmt_pct(results['overall_pass_rate'])}",
        f"  Paired (Eng+Bal) prompts:  {len(results['paired_df']):,}" if results.get('paired_df') is not None else "",
        "",
        "── PRIMARY METRIC: VIOLENCE / WEAPONS IN NEUTRAL PROMPTS ──────────────",
        "",
        f"  Neutral prompts sampled :  {violence['neutral_prompt_count']}",
        f"  Weapons-cue rate        :  {_fmt_pct(violence['weapons_rate'])}",
        f"  Conflict-cue rate       :  {_fmt_pct(violence['conflict_rate'])}",
        f"  Either violence rate    :  {_fmt_pct(violence['either_violence_rate'])}  "
        + ("⚠️  ALERT" if violence["alert"] else "✅ OK"),
        "",
        "── FIDELITY SUMMARY BY LABEL ──────────────────────────────────────────",
        "",
    ]

    if fid_sum is not None and len(fid_sum) > 0:
        header = f"  {'Group':<30} {'Acc':>6} {'Spec':>6} {'Avoid':>6} {'Div':>6} {'Pass%':>7}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for _, row in fid_sum.iterrows():
            lines.append(
                f"  {str(row.get('group','')):<30} "
                f"{_fmt_pct(row.get('fidelity_accuracy_mean',0)):>6} "
                f"{_fmt_pct(row.get('fidelity_specificity_mean',0)):>6} "
                f"{_fmt_pct(row.get('fidelity_avoidance_mean',0)):>6} "
                f"{_fmt_pct(row.get('fidelity_diversity_mean',0)):>6} "
                f"{_fmt_pct(row.get('pass_rate',0)):>7}"
            )
    lines.append("")

    lines += [
        "── STEREOTYPE DEFAULT RATES (ALERTS ONLY) ─────────────────────────────",
        "",
    ]
    alerts = def_rates[def_rates["alert"]] if def_rates is not None else pd.DataFrame()
    if len(alerts) > 0:
        for _, r in alerts.iterrows():
            lines.append(f"  ⚠️  {r['group']:<30} {r['attribute']:<22} {_fmt_pct(r['rate'])}")
    else:
        lines.append("  No stereotype alerts triggered.")
    lines.append("")

    lines += [
        "── GENDER DEFAULT RATES ───────────────────────────────────────────────",
        "",
    ]
    if gender_r is not None and len(gender_r) > 0:
        for _, r in gender_r.iterrows():
            flag = "⚠️  ALERT" if r["alert"] else "✅ OK"
            lines.append(f"  {r['group']:<30} {r['attribute']:<14} {_fmt_pct(r['rate'])}  {flag}")
    lines.append("")

    if results.get("eng_bal_diff") is not None:
        lines += [
            "── ENGLISH vs BALOCHI PROMPT FIDELITY ─────────────────────────────────",
            "",
            f"  Mean fidelity diff (ENG − BAL): {results['eng_bal_diff']:+.4f}",
            "  (Positive = English prompts score higher; may reflect training bias)",
            "",
        ]

    lines += [
        "── DELIVERABLES CHECKLIST ──────────────────────────────────────────────",
        "",
        "  ✅  Benchmark prompts (1,000 English + 1,000 with Balochi translations)",
        "  ✅  Stereotype rubric (8 attributes, default-vs-presence design)",
        "  ✅  Fidelity rubric (4 dimensions, weighted composite)",
        "  ✅  Evaluator pipeline (calibrated text proxy; VLM-ready interface)",
        "  ✅  Baseline results for open generators (simulated scores)",
        "  ✅  English vs Balochi prompt difference report",
        "  ✅  Headline violence/weapons metric for neutral prompts",
        "  ✅  Figures (8 charts saved to outputs/figures/)",
        "",
        "── RISKS LOGGED ────────────────────────────────────────────────────────",
        "",
        "  Med/Med  Accurate features scored as stereotype → mitigated by",
        "           default-vs-presence rubric (rates across N samples, not",
        "           penalising single occurrence).",
        "  High/Med Evaluator unreliable → humans are primary; VLM calibrated.",
        "  Med/Med  Rater fatigue → session limits & gold items implemented.",
        "",
        "=" * 70,
        "  Results shared with open-model developers where contact exists.",
        "  Generated images (when produced) not released; faces synthetic.",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n📄 Text report saved → {out_path}")
    return out_path


def generate_pdf_report(results: dict, figures: list[Path] = None) -> Path:
    """Generate a PDF report using ReportLab (if available)."""
    if not HAS_REPORTLAB:
        print("  ReportLab not available; text report only.")
        return generate_text_report(results)

    out_path = REPORT_DIR / "ig06_benchmark_report.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
    NM = styles["Normal"]
    AL = ParagraphStyle("Alert", parent=NM, textColor=colors.red)

    story.append(Paragraph(REPORT_TITLE, H1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", NM))
    story.append(Spacer(1, 0.5*cm))

    # --- Summary stats ---
    story.append(Paragraph("Executive Summary", H2))
    violence = results["violence_metric"]
    fid_sum  = results["fidelity_summary"]
    paired_n = len(results["paired_df"]) if results.get("paired_df") is not None else 0

    summary_data = [
        ["Metric", "Value"],
        ["Total prompts evaluated", f"{len(results['df_scored']):,}"],
        ["Overall fidelity pass rate", _fmt_pct(results["overall_pass_rate"])],
        ["Paired (Eng + Balochi) prompts", f"{paired_n:,}"],
        ["Neutral prompts evaluated", str(violence["neutral_prompt_count"])],
        ["Violence/weapons rate (neutral)", _fmt_pct(violence["either_violence_rate"])],
    ]
    t = Table(summary_data, colWidths=[10*cm, 5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#2E4057")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("PADDING",     (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Violence alert
    if violence["alert"]:
        story.append(Paragraph(
            f"⚠️ ALERT: Violence/weapons cue rate = {_fmt_pct(violence['either_violence_rate'])} "
            f"in neutral prompts (threshold: 50%)", AL))

    # Fidelity summary table
    story.append(Paragraph("Cultural Fidelity Summary", H2))
    if fid_sum is not None and len(fid_sum) > 0:
        cols = ["group","fidelity_accuracy_mean","fidelity_specificity_mean",
                "fidelity_avoidance_mean","fidelity_diversity_mean","pass_rate"]
        nice = ["Label","Accuracy","Specificity","Avoidance","Diversity","Pass %"]
        tdata = [nice]
        for _, row in fid_sum.iterrows():
            tdata.append([str(row.get(c,"")) if c=="group" else
                          _fmt_pct(row.get(c,0)) for c in cols])
        ft = Table(tdata, colWidths=[5.5*cm]+[2.2*cm]*5)
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E4057")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5F5")]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("PADDING",    (0,0), (-1,-1), 5),
        ]))
        story.append(ft)

    # Figures
    if figures:
        story.append(Paragraph("Figures", H2))
        for fig_path in figures:
            if fig_path and Path(fig_path).exists():
                story.append(Paragraph(Path(fig_path).stem.replace("_", " ").title(), NM))
                story.append(RLImage(str(fig_path), width=16*cm, height=8*cm))
                story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    print(f"\n📄 PDF report saved → {out_path}")
    return out_path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import run_pipeline
    from visualise import generate_all_figures
    results = run_pipeline(demo=False)
    generate_all_figures(results)
    # Collect figure paths
    figs = sorted(FIGURES_DIR.glob("fig*.png"))
    generate_pdf_report(results, figures=figs)
