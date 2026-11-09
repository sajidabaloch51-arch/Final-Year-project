"""
IG-06 · Main entry point — Full benchmark run
Run: python main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluate            import run_pipeline
from visualise           import generate_all_figures
from visualise_extended  import generate_extended_figures
from report_generator    import generate_text_report, generate_pdf_report
from calibrated_evaluator import train_calibrated_evaluator
from model_comparison    import run_model_comparison
from config              import FIGURES_DIR


def main():
    print("\n" + "=" * 60)
    print("  IG-06 · Balochistan T2I Cultural Benchmark")
    print("  Full Evaluation Run")
    print("=" * 60 + "\n")

    # 1. Core evaluation pipeline
    results = run_pipeline(demo=False)
    df      = results["df_scored"]

    # 2. Core figures (Fig 1–8)
    generate_all_figures(results)

    # 3. Train calibrated VLM evaluator
    print("\n── Calibrated Evaluator ────────────────────────────────────────────")
    eval_metrics = train_calibrated_evaluator(df)

    # 4. Multi-model comparison (n_seeds=4 for speed; spec: 16)
    print()
    model_results = run_model_comparison(df, n_seeds=4)

    # 5. Extended figures (Fig 9–16)
    generate_extended_figures(model_results, evaluator_metrics=eval_metrics)

    # 6. Reports (text + PDF with all figures)
    generate_text_report(results)
    figs = sorted(FIGURES_DIR.glob("fig*.png"))
    generate_pdf_report(results, figures=figs)

    print("\n" + "=" * 60)
    print("  ✅  IG-06 benchmark run complete!")
    print(f"  📁  Results  in: data/processed/")
    print(f"  📊  Figures  in: outputs/figures/  ({len(figs)} charts)")
    print(f"  🤖  Model    in: outputs/models/")
    print(f"  📄  Report   in: report/")
    print("=" * 60 + "\n")

    # Print brief headline summary
    vm = results["violence_metric"]
    es = eval_metrics
    ms = model_results["summary"]
    print("── HEADLINE RESULTS ────────────────────────────────────────────────")
    print(f"  Fidelity pass rate (prompts):      {results['overall_pass_rate']:.1%}")
    print(f"  Violence/weapons in neutral:        {vm['either_violence_rate']:.1%}")
    print(f"  Eng-Bal fidelity gap:              {results['eng_bal_diff']:+.4f}")
    print(f"  Calibrated evaluator accuracy:     {es['best_accuracy']:.3f} ({es['best_model']})")
    print(f"  Krippendorff's α (rule vs clf):    {es['krippendorff_alpha']:.4f}")
    print()
    print("  Model fidelity ranking:")
    for _, row in ms.sort_values("fidelity_mean", ascending=False).iterrows():
        alert_flag = "⚠️" if row.get("violence_neutral_rate", 0) >= 0.5 else "✅"
        print(f"    {alert_flag} {row['model']:<30} fidelity={row['fidelity_mean']:.3f}")
    print()


if __name__ == "__main__":
    main()
