import argparse
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BPR")
    parser.add_argument("--dataset", type=str, default="mylens")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config],
    )

if __name__ == "__main__":
    main()
