import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf

# -----------------------------
# 0. DOSYA YOLLARI
# -----------------------------
# Bu dosyanın olduğu klasör ( .../Startup-Success/src )
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Proje kökü ( .../Startup-Success )
BASE_DIR = os.path.dirname(THIS_DIR)

# CSV yolu ( .../Startup-Success/data/startups_clean.csv )
DATA_PATH = os.path.join(BASE_DIR, "data", "startups_clean.csv")

# Figürler klasörü ( .../Startup-Success/figures )
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("Using data file:", DATA_PATH)

# -----------------------------
# 1. VERİYİ YÜKLE
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("\n=== HEAD ===")
print(df.head())

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe())

print("\n=== INDUSTRY COUNTS ===")
print(df["industry"].value_counts())

print("\n=== OPERATING STATUS COUNTS ===")
print(df["operating_status"].value_counts())

# -----------------------------
# 2. GRAFİKLER (EDA)
# -----------------------------
sns.set(style="whitegrid")

# Funding distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["total_funding_usd"], bins=30)
plt.title("Funding Distribution")
plt.xlabel("Total funding (USD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "funding_distribution.png"))
plt.close()

# Funding by industry
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="industry", y="total_funding_usd")
plt.title("Funding by Industry")
plt.xlabel("Industry")
plt.ylabel("Total funding (USD)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "funding_by_industry.png"))
plt.close()

# Founding team size distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="num_founders")
plt.title("Founding Team Size Distribution")
plt.xlabel("Number of founders")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "founders_distribution.png"))
plt.close()

# -----------------------------
# 3. SUCCESS DEĞİŞKENİ
# -----------------------------
# Active = 1, diğer statüler = 0
df["success"] = df["operating_status"].apply(lambda x: 1 if x == "Active" else 0)

print("\n=== SUCCESS RATE ===")
print(df["success"].value_counts(normalize=True))

# -----------------------------
# 4. H1 – FUNDING vs SUCCESS (t-test)
# -----------------------------
success_funding = df[df["success"] == 1]["total_funding_usd"]
fail_funding    = df[df["success"] == 0]["total_funding_usd"]

print("\n=== H1: Funding vs Success (t-test) ===")
print("Mean funding (success=1):", success_funding.mean())
print("Mean funding (success=0):", fail_funding.mean())

t_stat, p_val = stats.ttest_ind(success_funding, fail_funding, equal_var=False)
print("t-stat:", t_stat)
print("p-value:", p_val)

# -----------------------------
# 5. H2 – FOUNDERS vs SUCCESS (t-test)
# -----------------------------
success_founders = df[df["success"] == 1]["num_founders"]
fail_founders    = df[df["success"] == 0]["num_founders"]

print("\n=== H2: Founders vs Success (t-test) ===")
print("Mean founders (success=1):", success_founders.mean())
print("Mean founders (success=0):", fail_founders.mean())

t2, p2 = stats.ttest_ind(success_founders, fail_founders, equal_var=False)
print("t-stat:", t2)
print("p-value:", p2)

# -----------------------------
# 6. H3 – LOGISTIC REGRESSION
# -----------------------------
df["log_funding"] = np.log1p(df["total_funding_usd"])

model = smf.logit("success ~ log_funding + num_founders + C(industry)", data=df).fit()

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(model.summary())

with open(os.path.join(FIG_DIR, "logit_summary.txt"), "w") as f:
    f.write(model.summary().as_text())

print("\nAnalysis finished. Figures saved in:", FIG_DIR)
