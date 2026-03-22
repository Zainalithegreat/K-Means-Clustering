import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

FILENAME = "QualityOfLifeDirty.csv"
df = pd.read_csv(FILENAME)

df_before = df.copy()

if "age_at_death" in df_before.columns:
    plt.figure()
    pd.to_numeric(df_before["age_at_death"], errors="coerce").hist(bins=20)
    plt.title("BEFORE CLEANING: Histogram - Age at Death")
    plt.xlabel("Age at Death")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if "avg_work_hours_per_day" in df_before.columns:
    plt.figure()
    plt.boxplot(pd.to_numeric(df_before["avg_work_hours_per_day"], errors="coerce").dropna())
    plt.title("BEFORE CLEANING: Box Plot - Work Hours per Day")
    plt.ylabel("Hours")
    plt.tight_layout()
    plt.show()

if ("avg_work_hours_per_day" in df_before.columns) and ("avg_sleep_hours_per_day" in df_before.columns):
    plt.figure()
    plt.scatter(
        pd.to_numeric(df_before["avg_work_hours_per_day"], errors="coerce"),
        pd.to_numeric(df_before["avg_sleep_hours_per_day"], errors="coerce"),
        alpha=0.6
    )
    plt.title("BEFORE CLEANING: Scatter - Work Hours vs Sleep Hours")
    plt.xlabel("Work Hours/Day")
    plt.ylabel("Sleep Hours/Day")
    plt.tight_layout()
    plt.show()

if ("gender" in df_before.columns) and ("avg_sleep_hours_per_day" in df_before.columns):
    plt.figure()
    groups = []
    labels = []
    for g in ["Male", "Female", "0"]:
        sub = df_before.loc[df_before["gender"] == g, "avg_sleep_hours_per_day"]
        sub = pd.to_numeric(sub, errors="coerce").dropna()
        if len(sub) > 0:
            groups.append(sub.values)
            labels.append(g)
    if len(groups) >= 2:
        plt.violinplot(groups, showmeans=True, showmedians=True)
        plt.title("BEFORE CLEANING: Violin Plot - Sleep Hours by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Sleep Hours/Day")
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.tight_layout()
        plt.show()

if "occupation_type" in df_before.columns:
    plt.figure(figsize=(10, 5))
    top_occ = df_before["occupation_type"].astype(str).value_counts()
    plt.bar(top_occ.index, top_occ.values)
    plt.title("BEFORE CLEANING: Bar Chart - Top Occupation Types")
    plt.xlabel("Occupation Type")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

print("\n======================")
print("STEP 0 - ORIGINAL DATA (BEFORE ANY CLEANING)")
print("======================")
print(df.head(10))
print("\nNull counts:\n", df.isnull().sum())
print("\nDtypes:\n", df.dtypes)

print("\n======================")
print("STEP 1 - DROP ID COLUMN")
print("======================")
print("BEFORE columns:", list(df.columns))
if "id" in df.columns:
    df = df.drop(columns=["id"])
print("AFTER columns:", list(df.columns))

print("\n======================")
print("STEP 2 - CLEAN CATEGORICAL COLUMNS (strip/case/fix invalids)")
print("======================")

cat_cols = []
for c in ["gender", "occupation_type"]:
    if c in df.columns:
        cat_cols.append(c)

for c in cat_cols:
    print(f"\n--- BEFORE cleaning '{c}' value counts (top 15) ---")
    print(df[c].value_counts(dropna=False).head(15))

    df[c] = df[c].astype(str).str.strip()
    df[c] = df[c].replace(
        ["", "nan", "NaN", "None", "NULL", "null", "N/A", "na", "-1", "0"],
        np.nan
    )
    df[c] = df[c].str.title()

    print(f"\n--- AFTER cleaning '{c}' value counts (top 15) ---")
    print(df[c].value_counts(dropna=False).head(15))

if "gender" in df.columns:
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
    valid_genders = {"Male", "Female", "Other"}
    df.loc[~df["gender"].isin(valid_genders) & df["gender"].notna(), "gender"] = np.nan
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

if "occupation_type" in df.columns:
    print("\n--- BEFORE removing invalid occupation_type rows (top 15) ---")
    print(df["occupation_type"].value_counts(dropna=False).head(15))
    df = df[df["occupation_type"].notna()]
    df = df[~df["occupation_type"].isin(["Banana"])]
    print("\n--- AFTER removing invalid occupation_type rows (top 15) ---")
    print(df["occupation_type"].value_counts(dropna=False).head(15))

print("\n======================")
print("STEP 3 - CLEAN NUMERIC COLUMNS (to_numeric, range checks, fill missing)")
print("======================")

numeric_cols = [
    "avg_work_hours_per_day",
    "avg_rest_hours_per_day",
    "avg_sleep_hours_per_day",
    "avg_exercise_hours_per_day",
    "age_at_death",
    "Happiness_Index_3highest"
]
numeric_cols = [c for c in numeric_cols if c in df.columns]

print("\n--- BEFORE numeric conversion: dtypes ---")
print(df[numeric_cols].dtypes)

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n--- AFTER numeric conversion: dtypes ---")
print(df[numeric_cols].dtypes)

range_rules = {
    "age_at_death": (0, 110),
    "avg_work_hours_per_day": (0, 24),
    "avg_rest_hours_per_day": (0, 24),
    "avg_sleep_hours_per_day": (0, 24),
    "avg_exercise_hours_per_day": (0, 24),
    "Happiness_Index_3highest": (0, 100)
}

for c, (low, high) in range_rules.items():
    if c in df.columns:
        before_bad = ((df[c] < low) | (df[c] > high)).sum()
        print(f"\n{c}: values outside [{low}, {high}] BEFORE = {before_bad}")
        df.loc[(df[c] < low) | (df[c] > high), c] = np.nan
        after_bad = ((df[c] < low) | (df[c] > high)).sum()
        print(f"{c}: values outside [{low}, {high}] AFTER = {after_bad}")
        df[c] = df[c].fillna(df[c].median())

print("\n--- Null counts AFTER numeric cleaning ---")
print(df.isnull().sum())

print("\n======================")
print("STEP 4 - REQUIRED VISUALS (5 DIFFERENT PLOT TYPES)")
print("======================")

if "age_at_death" in df.columns:
    plt.figure()
    df["age_at_death"].hist(bins=20)
    plt.title("Histogram: Age at Death (After Cleaning)")
    plt.xlabel("Age at Death")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if "avg_work_hours_per_day" in df.columns:
    plt.figure()
    plt.boxplot(df["avg_work_hours_per_day"].dropna(), vert=True)
    plt.title("Box Plot: Work Hours per Day (After Cleaning)")
    plt.ylabel("Hours")
    plt.tight_layout()
    plt.show()

if ("avg_work_hours_per_day" in df.columns) and ("avg_sleep_hours_per_day" in df.columns):
    plt.figure()
    plt.scatter(df["avg_work_hours_per_day"], df["avg_sleep_hours_per_day"], alpha=0.6)
    plt.title("Scatter: Work Hours vs Sleep Hours (After Cleaning)")
    plt.xlabel("Work Hours/Day")
    plt.ylabel("Sleep Hours/Day")
    plt.tight_layout()
    plt.show()

if ("gender" in df.columns) and ("avg_sleep_hours_per_day" in df.columns):
    plt.figure()
    groups = []
    labels = []
    for g in ["Male", "Female", "Other"]:
        sub = df.loc[df["gender"] == g, "avg_sleep_hours_per_day"].dropna()
        if len(sub) > 0:
            groups.append(sub.values)
            labels.append(g)
    if len(groups) >= 2:
        plt.violinplot(groups, showmeans=True, showmedians=True)
        plt.title("Violin Plot: Sleep Hours by Gender (After Cleaning)")
        plt.xlabel("Gender")
        plt.ylabel("Sleep Hours/Day")
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.tight_layout()
        plt.show()

if "occupation_type" in df.columns:
    plt.figure(figsize=(10, 5))
    top_occ = df["occupation_type"].value_counts()
    plt.bar(top_occ.index.astype(str), top_occ.values)
    plt.title("Bar Chart: Top 10 Occupation Types (After Cleaning)")
    plt.xlabel("Occupation Type")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

print("\n======================")
print("STEP 5 - OUTLIER HANDLING (IQR METHOD, REPLACE WITH MEDIAN)")
print("======================")

numeric_only = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_only:
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot BEFORE Outlier Handling: {col}")
    plt.tight_layout()
    plt.show()

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = outlier_mask.sum()
    print(f"{col}: outliers found = {outlier_count}")

    median = df[col].median()
    df.loc[outlier_mask, col] = median

    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Boxplot AFTER Outlier Handling: {col}")
    plt.tight_layout()
    plt.show()

print("\n======================")
print("STEP 6 - BUILD KMEANS-READY DATAFRAME + NORMALIZE")
print("======================")

kmeans_df = df.copy()

for c in ["gender", "occupation_type"]:
    if c in kmeans_df.columns:
        kmeans_df = pd.get_dummies(kmeans_df, columns=[c], drop_first=False)

if "Happiness_Index_3highest" in kmeans_df.columns:
    pass

kmeans_df = kmeans_df.select_dtypes(include=[np.number])

print("\n--- KMeans DF (BEFORE scaling) head(10) ---")
print(kmeans_df.head(10))

scaler = MinMaxScaler()
kmeans_scaled = pd.DataFrame(
    scaler.fit_transform(kmeans_df),
    columns=kmeans_df.columns
)

print("\n--- KMeans DF (AFTER MinMax scaling) head(10) ---")
print(kmeans_scaled.head(10))

print("\n======================")
print("STEP 7 - SAVE CLEANED CSVs")
print("======================")

CLEANED_OUT = "ZainAli_QualityOfLife_Cleaned.csv"
KMEANS_OUT = "ZainAli_QualityOfLife_KMeansReady_Scaled.csv"

df.to_csv(CLEANED_OUT, index=False)
kmeans_scaled.to_csv(KMEANS_OUT, index=False)

print(f"Saved cleaned dataset: {CLEANED_OUT}")
print(f"Saved KMeans-ready scaled dataset: {KMEANS_OUT}")

print("\n======================")
print("FINAL CHECKS")
print("======================")
print("Final cleaned df head(10):")
print(df.head(10))
print("\nFinal null counts:\n", df.isnull().sum())
print("\nFinal dtypes:\n", df.dtypes)

print("\n======================")
print("PART 2 - KMEANS FORMATTING")
print("======================")

print("\nDtypes BEFORE removing qualitative columns:")
print(df.dtypes)

qualitative_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("\nQualitative columns identified:", qualitative_cols)

df_kmeans = df.drop(columns=qualitative_cols)

print("\nDtypes AFTER removing qualitative columns:")
print(df_kmeans.dtypes)

print("\nKMeans-ready dataset preview:")
print(df_kmeans.head(10))

OUTPUT_KMEANS = "ZainAli_KMeansReady.csv"
df_kmeans.to_csv(OUTPUT_KMEANS, index=False)
print(f"\nSaved KMeans-ready dataset: {OUTPUT_KMEANS}")

print("\n======================")
print("PART 3 - READ CLEANED KMEANS DATASET")
print("======================")

KMEANS_FILE = "ZainAli_KMeansReady.csv"
df_kmeans = pd.read_csv(KMEANS_FILE)

print("\n--- KMeans-ready dataframe (head 15) ---")
print(df_kmeans.head(15))
print("\nShape:", df_kmeans.shape)
print("\nDtypes:\n", df_kmeans.dtypes)

candidate_3d = ["avg_work_hours_per_day", "avg_sleep_hours_per_day", "age_at_death"]
vars_3d = [c for c in candidate_3d if c in df_kmeans.columns]
if len(vars_3d) < 3:
    vars_3d = df_kmeans.columns[:3].tolist()

x_col, y_col, z_col = vars_3d[0], vars_3d[1], vars_3d[2]
print(f"\n3D plot variables selected: {x_col}, {y_col}, {z_col}")

def run_kmeans_and_report(k: int, df_data: pd.DataFrame, x_col: str, y_col: str, z_col: str):
    print("\n========================================")
    print(f"KMEANS RESULTS (k = {k})")
    print("========================================")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df_data)

    print("\nCluster labels (first 30 rows):")
    print(labels[:30])

    labeled_df = df_data.copy()
    labeled_df["cluster"] = labels

    print("\nData with cluster labels (head 15):")
    print(labeled_df.head(15))

    centers_df = pd.DataFrame(model.cluster_centers_, columns=df_data.columns)

    print("\nCluster Centers (Centroids):")
    print(centers_df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        labeled_df[x_col],
        labeled_df[y_col],
        labeled_df[z_col],
        c=labeled_df["cluster"],
        alpha=0.7
    )

    ax.set_title(f"3D Scatter Plot (k={k}) using {x_col}, {y_col}, {z_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    plt.tight_layout()
    plt.show()

    return labeled_df, centers_df

labeled_k2, centers_k2 = run_kmeans_and_report(2, df_kmeans, x_col, y_col, z_col)
labeled_k3, centers_k3 = run_kmeans_and_report(3, df_kmeans, x_col, y_col, z_col)

labeled_k2.to_csv("ZainAli_KMeans_k2_Labeled.csv", index=False)
labeled_k3.to_csv("ZainAli_KMeans_k3_Labeled.csv", index=False)
print("\nSaved labeled cluster outputs: ZainAli_KMeans_k2_Labeled.csv and ZainAli_KMeans_k3_Labeled.csv")

new_person = np.array([5.0, 3.33, 9.0, 4.44, 77.0])

cols_5 = [
    "avg_work_hours_per_day",
    "avg_rest_hours_per_day",
    "avg_sleep_hours_per_day",
    "avg_exercise_hours_per_day",
    "age_at_death"
]

df_clean = pd.read_csv("ZainAli_QualityOfLife_Cleaned.csv")
df_5 = df_clean[cols_5].copy()

def distances_to_centroids(k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(df_5)

    centroids = model.cluster_centers_

    print("\n========================================")
    print(f"PART 4 - Distances to centroids (k={k})")
    print("========================================")

    print("\nCentroids (in the same order as cols_5):")
    print(pd.DataFrame(centroids, columns=cols_5))

    distances = []
    for i, c in enumerate(centroids):
        diff = new_person - c
        sq = diff ** 2
        dist = np.sqrt(np.sum(sq))
        distances.append(dist)

        print(f"\nCluster {i} distance calculation:")
        print("diff =", diff)
        print("squared =", sq)
        print("distance =", dist)

    closest = int(np.argmin(distances))
    print(f"\nClosest cluster for k={k}: Cluster {closest}")
    print("Distances:", distances)

distances_to_centroids(2)
distances_to_centroids(3)

print("\n======================")
print("PART 5 - DISCRETIZATION + LABEL CREATION")
print("======================")

df5 = pd.read_csv("ZainAli_QualityOfLife_Cleaned.csv")
df5 = df5[cols_5].copy()

print("\n--- Clean quantitative dataset (head 15) ---")
print(df5.head(15))

bins = [-np.inf, 59, 69, 79, np.inf]
labels_age = ["VeryYoung", "Young", "Aged", "Older"]

df5["Age_Label"] = pd.cut(df5["age_at_death"], bins=bins, labels=labels_age)

print("\n--- Labeled dataset (head 20) ---")
print(df5.head(20))

print("\nAge_Label counts:")
print(df5["Age_Label"].value_counts())

LABELED_OUT = "ZainAli_QualityOfLife_Labeled.csv"
df5.to_csv(LABELED_OUT, index=False)
print(f"\nSaved labeled dataset: {LABELED_OUT}")

X5 = df5[cols_5].copy()

print("\n======================")
print("PART 5 - KMEANS ON QUANTITATIVE DATA ONLY (k=4)")
print("======================")

k4_model = KMeans(n_clusters=4, random_state=42, n_init=10)
k4_labels = k4_model.fit_predict(X5)

df5_with_clusters = df5.copy()
df5_with_clusters["Cluster_k4"] = k4_labels

print("\n--- Data with Age_Label and Cluster_k4 (head 20) ---")
print(df5_with_clusters.head(20))

centroids_k4 = pd.DataFrame(k4_model.cluster_centers_, columns=cols_5)

print("\n--- Cluster Centroids for k=4 (each centroid has 5 values) ---")
print(centroids_k4)

centroids_k4.to_csv("ZainAli_k4_Centroids.csv", index=False)


ct = pd.crosstab(df5_with_clusters["Age_Label"], df5_with_clusters["Cluster_k4"])
print("\nCrosstab (Age_Label rows vs Cluster columns):")
print(ct)

row_pct = ct.div(ct.sum(axis=1), axis=0).round(3)
print("\nRow percentages (within each Age_Label):")
print(row_pct)

ct.to_csv("ZainAli_AgeLabel_vs_Cluster_k4_Crosstab.csv")
row_pct.to_csv("ZainAli_AgeLabel_vs_Cluster_k4_RowPct.csv")
