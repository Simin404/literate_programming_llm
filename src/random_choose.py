import argparse
import random
import pandas as pd

from utils import *  # if you need side effects/utilities from here
from prediction import codenet_data_analysis


def run_experiments(
    train_path: str,
    test_path: str,
    num_task_list=None,
    top_n: int = 30,
    repeats: int = 3,
    seed: int | None = None,
):
    if num_task_list is None:
        num_task_list = [5]

    if seed is not None:
        random.seed(seed)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare candidate pools
    all_tasks = train_df["task"].value_counts().nlargest(top_n).index.tolist()
    all_langs = train_df["language"].value_counts().nlargest(top_n).index.tolist()

    # Prepare storage for chosen combinations
    chosen_dic = {n: {"chosen_lang": [], "chosen_task": []} for n in num_task_list}

    # Run experiments
    for n in num_task_list:
        for _ in range(repeats):
            chosen_task = random.choices(all_tasks, k=n)
            chosen_lang = random.choices(all_langs, k=n)

            chosen_dic[n]["chosen_task"].append(chosen_task)
            chosen_dic[n]["chosen_lang"].append(chosen_lang)

            results = codenet_data_analysis(chosen_task, chosen_lang, train_df, test_df)
            print(results)
            print("=" * 56)

    return chosen_dic


def main():
    parser = argparse.ArgumentParser(description="Run CodeNet data analysis experiments.")
    parser.add_argument(
        "--train-path",
        default="codenet/train_1725993.csv",
        help="Path to training CSV (default: codenet/train_1725993.csv)",
    )
    parser.add_argument(
        "--test-path",
        default="codenet/test_1824.csv",
        help="Path to test CSV (default: codenet/test_1824.csv)",
    )
    parser.add_argument(
        "--num-task",
        nargs="+",
        type=int,
        default=[10],
        help="List of counts for how many tasks/langs to pick each run (default: 10). Example: --num-task 10 20 30",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Select from the top-N most frequent tasks/langs (default: 30)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many repetitions per 'num-task' value (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()
    print(f"Number of tasks: {args.num_task}, Repeat times: {args.repeats}")
    run_experiments(
        train_path=args.train_path,
        test_path=args.test_path,
        num_task_list=args.num_task,
        top_n=args.top_n,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
