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

    print("\n=== Founder Count Summary ===")
    print(df["founder_count"].value_counts())

    print("\n=== GDP Growth Summary ===")
    print(df["gdp_growth"].describe())


def plot_distributions(df):

    # 1) Histogram of funding
    plt.figure()
    df["total_funding_usd"].hist(bins=30)
    plt.title("Funding Distribution")
    plt.xlabel("Funding (USD)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("../graphs/funding_distribution.png")
    plt.close()

    # 2) Funding vs Status boxplot
    plt.figure()
    sns.boxplot(data=df, x="status", y="total_funding_usd")
    plt.title("Funding by Status")
    plt.tight_layout()
    plt.savefig("../graphs/funding_by_status.png")
    plt.close()

    # 3) Founder experience vs Status boxplot
    plt.figure()
    sns.boxplot(data=df, x="status", y="founder_experience_years")
    plt.title("Founder Experience by Status")
    plt.tight_layout()
    plt.savefig("../graphs/experience_by_status.png")
    plt.close()

    # 4) GDP growth vs funding scatter plot (no hypothesis test yet)
    plt.figure()
    plt.scatter(df["gdp_growth"], df["total_funding_usd"])
    plt.xlabel("GDP Growth (%)")
    plt.ylabel("Total Funding (USD)")
    plt.title("GDP Growth vs Startup Funding")
    plt.tight_layout()
    plt.savefig("../graphs/gdp_vs_funding.png")
    plt.close()

    # 5) Success rate by founder count (your "2")
    temp = df.copy()
    temp["is_success"] = (temp["status"] == "Success").astype(int)
    rate_by_founders = temp.groupby("founder_count")["is_success"].mean().reset_index()

    plt.figure()
    plt.bar(rate_by_founders["founder_count"], rate_by_founders["is_success"])
    plt.xlabel("Founder Count")
    plt.ylabel("Success Rate")
    plt.title("Success Rate by Founder Count")
    plt.tight_layout()
    plt.savefig("../graphs/success_rate_by_founder_count.png")
    plt.close()

    # 6) Success rate by GDP growth (binned categories) 
    gdp_bins = [0, 2, 4, 8]
    gdp_labels = ["Low (0–2%)", "Medium (2–4%)", "High (4–8%)"]
    temp["gdp_bin"] = pd.cut(
        temp["gdp_growth"],
        bins=gdp_bins,
        labels=gdp_labels,
        include_lowest=True
    )
    rate_by_gdp = temp.groupby("gdp_bin")["is_success"].mean().reset_index()

    plt.figure()
    plt.bar(rate_by_gdp["gdp_bin"], rate_by_gdp["is_success"])
    plt.xlabel("GDP Growth Category")
    plt.ylabel("Success Rate")
    plt.title("Success Rate by GDP Growth Category")
    plt.tight_layout()
    plt.savefig("../graphs/success_rate_by_gdp_growth.png")
    plt.close()

    # 7) Sector vs Status countplot 
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="sector", hue="status")
    plt.xticks(rotation=45, ha="right")
    plt.title("Sector vs Status")
    plt.tight_layout()
    plt.savefig("../graphs/sector_vs_status.png")
    plt.close()

   

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


def test_founders_2_3_vs_solo(df):
    """Compare performance of startups with 2–3 founders vs solo founders."""

    df = df.dropna(subset=["founder_count"])

    solo = df[df["founder_count"] == 1]
    small_team = df[(df["founder_count"] == 2) | (df["founder_count"] == 3)]

    solo_funding = solo["total_funding_usd"].dropna()
    team_funding = small_team["total_funding_usd"].dropna()

    t_stat, p_value = stats.ttest_ind(team_funding, solo_funding, equal_var=False)

    print("\n=== 2–3 Founders vs Solo Founders ===")
    print("Mean funding (solo founder):", solo_funding.mean())
    print("Mean funding (2–3 founders):", team_funding.mean())
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    solo_success_rate = (solo["status"] == "Success").mean()
    team_success_rate = (small_team["status"] == "Success").mean()

    print("Success rate (solo founder):", solo_success_rate)
    print("Success rate (2–3 founders):", team_success_rate)


def main():
    df = load_data()
    basic_eda(df)
    plot_distributions(df)
    test_funding_vs_status(df)
    test_founders_2_3_vs_solo(df)


if __name__ == "__main__":
    main()
