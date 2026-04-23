import argparse

from src.core.domain_ablation_pipeline import run_from_csv


def main():
    parser = argparse.ArgumentParser(description="Run extracted domain feature-engineering ablation pipeline")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--out", default="outputs/domain_ablation", help="Output directory")
    args = parser.parse_args()

    results = run_from_csv(args.train, args.test, args.out)
    print("Run completed")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
