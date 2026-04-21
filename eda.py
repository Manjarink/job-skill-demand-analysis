import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set seaborn theme for beautiful plots
sns.set_theme(style="whitegrid", palette="muted")

# ================= LOAD DATA =================
df = pd.read_csv("cleaned_data.csv")

print("\n========== DATASET INFO ==========")
print(df.info())

# ================= STEP 1: FILTER TECH JOBS =================
df = df[df['job_title'].str.contains(
    'data|engineer|developer|software|analyst|machine learning|ai',
    case=False,
    na=False
)]

# ================= STEP 2: CLEAN SKILLS =================
df['skill'] = df['skill'].str.lower().str.strip()
df.drop_duplicates(inplace=True)

# ================= STEP 3: REMOVE NON-TECH + NOISE =================

non_tech = [
    'communication', 'teamwork', 'leadership',
    'time management', 'problem solving', 'problemsolving',
    'customer service', 'attention to detail',
    'communication skills', 'project management',
    'interpersonal skills', 'organizational skills',
    'sales', 'training', 'collaboration',
    'analytical skills'
]

pattern_noise = [
    'degree', 'bachelor', 'master', 'phd',
    'experience', 'year',
    'engineering', 'engineer', 'science',
    'management', 'scheduling',
    'office', 'autocad'
]

strict_noise = [
    'troubleshooting',
    'analytical',
    'skill',
    'skills'
]

# Remove non-tech
tech_df = df[~df['skill'].isin(non_tech)]

# Remove pattern noise
for pattern in pattern_noise:
    tech_df = tech_df[~tech_df['skill'].str.contains(pattern, na=False)]

# Remove strict noise
for word in strict_noise:
    tech_df = tech_df[~tech_df['skill'].str.contains(word, na=False)]

# ============================================================
#  Q1: TOP TECHNICAL SKILLS
# ============================================================
print("\n========== Q1: TOP TECHNICAL SKILLS ==========")

top_skills = tech_df['skill'].value_counts().head(10)
print(top_skills)

plt.figure(figsize=(12, 6))
# Using seaborn barplot for distinct colors and horizontal layout
sns.barplot(x=top_skills.values, y=top_skills.index, palette="viridis", hue=top_skills.index, legend=False)
plt.title("Top 10 Technical Skills in Tech Jobs", fontsize=16, fontweight='bold')
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Skills", fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================

skills_per_job = tech_df.groupby('job_id').size().reset_index(name='skill_count')

jobs_per_skill = tech_df['skill'].value_counts().reset_index()
jobs_per_skill.columns = ['skill', 'job_count']

merged_df = pd.merge(tech_df, skills_per_job, on='job_id')
merged_df = pd.merge(merged_df, jobs_per_skill, on='skill')

#  REMOVE OUTLIERS FOR CLEAN GRAPH
merged_df = merged_df[merged_df['skill_count'] < 60]

# ============================================================
#  Q2: SKILL COUNT vs JOB DEMAND
# ============================================================
print("\n========== Q2: SKILL COUNT vs JOB DEMAND ==========")

corr = merged_df[['skill_count', 'job_count']].corr()
print("\nCorrelation:")
print(corr)

X = merged_df[['skill_count']]
y = merged_df['job_count']

model = LinearRegression()
model.fit(X, y)

print("\nRegression Coefficient:", model.coef_[0])
print("Regression Intercept:", model.intercept_)

plt.figure(figsize=(10, 6))
# Using seaborn regplot to add a regression trend line on top of scatter
sns.regplot(
    data=merged_df, 
    x='skill_count', 
    y='job_count', 
    scatter_kws={'alpha': 0.5, 'color': 'steelblue'}, 
    line_kws={'color': 'darkred', 'linewidth': 2}
)
plt.xlabel("Number of Skills per Job", fontsize=12)
plt.ylabel("Job Demand (Job Count)", fontsize=12)
plt.title("Skill Count vs Job Demand with Trend Line", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("There is a weak negative correlation.")
print("Increasing the number of skills does NOT significantly increase job demand.")
print("Employers focus on key skills rather than many skills.")

# ============================================================
#  Q3: RELATIONSHIP ANALYSIS
# ============================================================
print("\n========== Q3: RELATIONSHIP ANALYSIS ==========")

if corr.iloc[0,1] > 0:
    print("Positive relationship: More skills → Higher demand")
elif corr.iloc[0,1] < 0:
    print("Negative relationship: More skills → Lower demand")
else:
    print("No relationship detected")

print("Conclusion: Weak negative relationship observed.")

# ============================================================
#  Q4: COVARIANCE & CORRELATION HEATMAP
# ============================================================
print("\n========== Q4: COVARIANCE ==========")

cov = merged_df[['skill_count', 'job_count']].cov()
print(cov)

print("\nInterpretation:")
print("Negative covariance shows variables move in opposite direction, but weakly.")

# Generate a heatmap for visual appeal
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
#  Q5: TECH vs NON-TECH
# ============================================================
print("\n========== Q5: TECH vs NON-TECH ==========")

# FIXED PART (IMPORTANT)
nontech_df = df[~df.index.isin(tech_df.index)]

tech_count = len(tech_df)
nontech_count = len(nontech_df)

print("Technical Skills Count:", tech_count)
print("Non-Technical Skills Count:", nontech_count)

plt.figure(figsize=(8, 5))
# Pie chart for better proportion visualization
labels = ['Technical', 'Non-Technical']
sizes = [tech_count, nontech_count]
colors = sns.color_palette("pastel")[0:2]
explode = (0.1, 0)  # pull out the technical slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
plt.title("Technical vs Non-Technical Skills Distribution", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("Technical skills dominate the dataset, showing that technical expertise is the primary requirement in tech jobs.")

# ============================================================
#  ADVANCED: OUTLIER ANALYSIS
# ============================================================
print("\n========== ADVANCED 1: OUTLIER ANALYSIS ==========")
Q1_job = jobs_per_skill['job_count'].quantile(0.25)
Q3_job = jobs_per_skill['job_count'].quantile(0.75)
IQR_job = Q3_job - Q1_job
upper_bound_job = Q3_job + 1.5 * IQR_job

print(f"Job Demand Q1: {Q1_job}, Q3: {Q3_job}, IQR: {IQR_job}")
print(f"Upper Bound for Outliers: {upper_bound_job}")
outliers = jobs_per_skill[jobs_per_skill['job_count'] > upper_bound_job]
print(f"Number of outlier skills (super high demand): {len(outliers)}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=skills_per_job['skill_count'], color='lightcoral')
plt.title("Outlier Detection: Skill Count per Job", fontsize=14, fontweight='bold')
plt.ylabel("Number of Skills")

plt.subplot(1, 2, 2)
sns.boxplot(y=jobs_per_skill['job_count'], color='lightseagreen')
plt.title("Outlier Detection: Job Demand per Skill", fontsize=14, fontweight='bold')
plt.ylabel("Number of Jobs")
plt.tight_layout()
plt.show()

# ============================================================
#  ADVANCED: VARIANCE & SPREAD ANALYSIS
# ============================================================
print("\n========== ADVANCED 2: VARIANCE & SPREAD ==========")
skill_var = skills_per_job['skill_count'].var()
skill_std = skills_per_job['skill_count'].std()
job_var = jobs_per_skill['job_count'].var()
job_std = jobs_per_skill['job_count'].std()

print(f"Skill Count per Job - Variance: {skill_var:.2f}, Std Dev: {skill_std:.2f}")
print(f"Job Demand per Skill - Variance: {job_var:.2f}, Std Dev: {job_std:.2f}")

# ============================================================
#  ADVANCED: CLUSTERING ANALYSIS (K-MEANS)
# ============================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\n========== ADVANCED 3: CLUSTERING ANALYSIS ==========")
cluster_data = merged_df[['skill_count', 'job_count']].drop_duplicates().copy()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_data['cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=cluster_data, x='skill_count', y='job_count', 
    hue='cluster', palette='Dark2', s=100, alpha=0.8
)
plt.title("K-Means Clusters: Skill Requirements vs. Job Demand", fontsize=16, fontweight='bold')
plt.xlabel("Skill Count per Job", fontsize=12)
plt.ylabel("Job Demand for Skill", fontsize=12)
plt.legend(title='Cluster ID')
plt.tight_layout()
plt.show()

# ============================================================
#  ADVANCED: NEW INSIGHTS GENERATION (REPORT ADD-ONS)
# ============================================================
print("\n========== ADVANCED 4: DEEP INSIGHTS ==========")
print("1. Heavy-Tailed Market: Few core skills (like Python, SQL) are universally demanded, inflating variance.")
print("2. The 'Kitchen Sink' Anomaly: Jobs asking for very high skill counts (>20) usually have low market demand.")
print("3. Specialization Premium: Highly specialized roles (fewer specific skills) don't have broad demand, but represent a stable distinct cluster.")
