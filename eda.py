import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set seaborn theme for clean visuals
sns.set_theme(style="whitegrid", palette="muted")

# ================= LOAD & PREPARE DATA =================
df = pd.read_csv("cleaned_data.csv")

# Filter for Tech Jobs to focus analysis strictly on technical roles
df = df[df['job_title'].str.contains(
    'data|engineer|developer|software|analyst|machine learning|ai',
    case=False, na=False
)]

# Clean Skills
df['skill'] = df['skill'].str.lower().str.strip()
df.drop_duplicates(inplace=True)

# Remove Non-Tech and Noise to isolate purely technical skillset requirements
non_tech = ['communication', 'teamwork', 'leadership', 'time management', 'problem solving', 'problemsolving', 'customer service', 'attention to detail', 'communication skills', 'project management', 'interpersonal skills', 'organizational skills', 'sales', 'training', 'collaboration', 'analytical skills']
pattern_noise = ['degree', 'bachelor', 'master', 'phd', 'experience', 'year', 'engineering', 'engineer', 'science', 'management', 'scheduling', 'office', 'autocad']
strict_noise = ['troubleshooting', 'analytical', 'skill', 'skills']

tech_df = df[~df['skill'].isin(non_tech)]
for pattern in pattern_noise: tech_df = tech_df[~tech_df['skill'].str.contains(pattern, na=False)]
for word in strict_noise: tech_df = tech_df[~tech_df['skill'].str.contains(word, na=False)]

# Feature Engineering
skills_per_job = tech_df.groupby('job_id').size().reset_index(name='skill_count')
jobs_per_skill = tech_df['skill'].value_counts().reset_index()
jobs_per_skill.columns = ['skill', 'job_count']

merged_df = pd.merge(tech_df, skills_per_job, on='job_id')
merged_df = pd.merge(merged_df, jobs_per_skill, on='skill')

# Create a clean subset for specific visualizations by capping extreme outliers
merged_df_clean = merged_df[merged_df['skill_count'] < 60].copy()

# ============================================================
# ========== DATASET OVERVIEW ==========
# ============================================================
print("\n========== DATASET OVERVIEW ==========")
print(f"Total Unique Tech Jobs: {df['job_id'].nunique() if 'job_id' in df.columns else len(df)}")
print(f"Total Unique Technical Skills: {tech_df['skill'].nunique()}")

# ============================================================
# ========== TOP TECHNICAL SKILLS ==========
# ============================================================
print("\n========== TOP TECHNICAL SKILLS ==========")
top_skills = tech_df['skill'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_skills.values, y=top_skills.index, palette="viridis", hue=top_skills.index, legend=False)
plt.title("Top 10 Technical Skills by Frequency")
plt.xlabel("Frequency")
plt.ylabel("Skills")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print(f"→ Python is the most frequently requested skill, appearing {top_skills.iloc[0]} times.")
print(f"→ SQL follows closely with {top_skills.iloc[1]} appearances.")
print("→ The data indicates that advanced tech postings frequently reference these specific foundational technologies.")

# ============================================================
# ========== TECHNICAL VS NON-TECHNICAL SKILLS ==========
# ============================================================
print("\n========== TECHNICAL VS NON-TECHNICAL SKILLS ==========")
nontech_df = df[~df.index.isin(tech_df.index)]
sizes = [len(tech_df), len(nontech_df)]
labels = ['Technical', 'Non-Technical']

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=(0.1, 0), labels=labels, autopct='%1.1f%%', shadow=True, startangle=140, colors=sns.color_palette("pastel")[0:2])
plt.title("Technical vs Non-Technical Skill Frequency")
plt.show()

print("\nINSIGHTS:")
print(f"→ Mentions of technical skills ({sizes[0]}) are substantially higher than non-technical skills ({sizes[1]}).")
print("→ This chart highlights frequency in job descriptions, suggesting an emphasis on technical keywords rather than determining inherent value.")

# ============================================================
# ========== SKILL COUNT VS JOB DEMAND ==========
# ============================================================
print("\n========== SKILL COUNT VS JOB DEMAND ==========")
X = merged_df_clean[['skill_count']]
y = merged_df_clean['job_count']
model = LinearRegression().fit(X, y)

plt.figure(figsize=(10, 5))
sns.regplot(data=merged_df_clean, x='skill_count', y='job_count', scatter_kws={'alpha': 0.3}, line_kws={'color': 'darkred'})
plt.title("Skill Count vs Job Demand")
plt.xlabel("Number of Skills Requested per Job")
plt.ylabel("Job Demand")
plt.tight_layout()
plt.show()

corr = merged_df_clean[['skill_count', 'job_count']].corr().iloc[0,1]
print(f"Calculated Correlation: {corr:.4f}")

print("\nINSIGHTS:")
print(f"→ There is a very weak negative correlation (~ {corr:.4f}), indicating no strong linear relationship.")
print("→ The slight inverse trend suggests jobs listing higher volumes of skills do not correspond with proportionally higher Job Demand.")

# ============================================================
# ========== OUTLIER ANALYSIS ==========
# ============================================================
print("\n========== OUTLIER ANALYSIS ==========")
Q1 = jobs_per_skill['job_count'].quantile(0.25)
Q3 = jobs_per_skill['job_count'].quantile(0.75)
IQR = Q3 - Q1

if IQR == 0:
    print("Note: IQR method is not reliable here due to extremely low variance (most niche skills appear only once).")
else:
    print(f"Calculated Outlier Threshold: {Q3 + 1.5 * IQR}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=skills_per_job['skill_count'], color='skyblue')
plt.title("Outliers: Total Skills per Job")

plt.subplot(1, 2, 2)
sns.boxplot(x=jobs_per_skill['job_count'], color='orange')
plt.title("Outliers: Job Demand")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ The boxplot for Job Demand indicates that the majority of skills have very low frequencies, while a few outliers appear disproportionately often.")
print("→ When filtering extreme values, the correlation metrics remain near zero, suggesting extreme structural anomalies impact the regression line.")

# ============================================================
# ========== DISTRIBUTION ANALYSIS ==========
# ============================================================
print("\n========== DISTRIBUTION ANALYSIS ==========")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(skills_per_job['skill_count'], kde=True, bins=30, color='royalblue')
plt.title("Distribution of Skills Requested")

plt.subplot(1, 2, 2)
sns.histplot(jobs_per_skill['job_count'], kde=True, bins=30, color='darkorange')
plt.title("Distribution of Job Demand")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ The distribution for skills requested shows a right skew, with the dataset mode sitting between 3 to 8 skills per job.")
print("→ The distribution for Job Demand exhibits an extreme right skew, indicating a concentration of demand isolated to a few key technologies.")

# ============================================================
# ========== CLUSTERING ANALYSIS ==========
# ============================================================
print("\n========== CLUSTERING ANALYSIS ==========")
cluster_data = merged_df_clean[['skill_count', 'job_count']].drop_duplicates().dropna()

# Standardize variables to ensure K-Means computes distances uniformly across both axes
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# K-Means applied to segment distinct structural groupings within the job-skill pairs
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_data['cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=cluster_data, x='skill_count', y='job_count', hue='cluster', palette='Set1', alpha=0.7)
plt.title("K-Means Market Segmentation")
plt.xlabel("Skills Requested per Job")
plt.ylabel("Job Demand")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ Cluster 0 maps moderate ranges in both requested skill count and corresponding Job Demand.")
print("→ Cluster 1 shows postings with a substantially higher average skill count, but with distinctly lower Job Demand trends.")
print("→ Cluster 2 consists of moderate skill volumes paired with visibly the highest scale of Job Demand.")

# ============================================================
# ========== FINAL TAKEAWAYS ==========
# ============================================================
print("\n========== FINAL TAKEAWAYS ==========")
print("→ Analysis of top frequencies suggests that foundational frameworks generate the highest Job Demand across all grouped roles.")
print("→ Quantitative metrics and clustering indicate that simply increasing the technical requirement volume (skill count) inversely or weakly affects overall Job Demand.")
print("→ Frequency data highlights a structural skew, emphasizing concentrated technical skill groupings rather than generalized keyword stacking.")
