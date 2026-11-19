"""Fusionne les annotations des deux membres -> data/eval/labels.csv (ground truth figé).
Pertinent = les DEUX annotateurs ont mis 1. Les désaccords sont listés pour discussion."""
from cvmatch.config import get_settings
from cvmatch.evaluation import finalize_labels


def main() -> None:
    s = get_settings()
    stats = finalize_labels(
        s.eval_dir / "pool_to_annotate.csv",
        s.eval_dir / "labels.csv",
        s.eval_dir / "disagreements.csv",
    )
    print("✔ labels.csv écrit")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Rappel : figez labels.csv AVANT les derniers tests ; ne réglez rien dessus.")


if __name__ == "__main__":
    main()
