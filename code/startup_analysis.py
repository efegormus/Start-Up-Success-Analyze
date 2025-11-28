import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(path="../data/startup_dataset.csv"):
    """Load the startup dataset."""
    df = pd.read_csv(path)
    return df


def basic_eda(df):
    """Print basic exploratory data analysis outputs."""
    print("\n=== Head (first 5 rows) ===")
    print(df.head())

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Column Types ===")
    print(df.dtypes)

    print("\n=== Numerical Summary ===")
    print(df.describe())

    print("\n=== Status Distribution ===")
    print(df["status"].value_counts())


def plot_distributions(df):
    """Generate plots and save them into the graphs folder."""

    # Histogram of Funding
    plt.figure()
    df["total_funding_usd"].hist(bins=30)
    plt.title("Funding Distribution")
    plt.xlabel("Funding (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("../graphs/funding_distribution.png")
    plt.close()

    # Boxplot: Funding vs Status
    plt.figure()
    sns.boxplot(data=df, x="status", y="total_funding_usd")
    plt.title("Funding by Status")
    plt.tight_layout()
    plt.savefig("../graphs/funding_by_status.png")
    plt.close()

    # Boxplot: Founder Experience vs Status
    plt.figure()
    sns.boxplot(data=df, x="status", y="founder_experience_years")
    plt.title("Founder Experience by Status")
    plt.tight_layout()
    plt.savefig("../graphs/experience_by_status.png")
    plt.close()

    print("\nPlots have been saved into the '../graphs' folder.")


def test_funding_vs_status(df):
    """Independent t-test comparing funding of successful and failed startups."""
    success = df[df["status"] == "Success"]["total_funding_usd"]
    failure = df[df["status"] == "Failure"]["total_funding_usd"]

    t_stat, p_value = stats.ttest_ind(success, failure, equal_var=False)

    print("\n=== T-test: Funding vs Status ===")
    print("Mean funding (Success):", success.mean())
    print("Mean funding (Failure):", failure.mean())
    print("t-statistic:", t_stat)
    print("p-value:", p_value)


def main():
    df = load_data()
    basic_eda(df)
    plot_distributions(df)
    test_funding_vs_status(df)


if __name__ == "__main__":
    main()
