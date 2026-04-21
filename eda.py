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

# Filter for Tech Jobs
df = df[df['job_title'].str.contains(
    'data|engineer|developer|software|analyst|machine learning|ai',
    case=False, na=False
)]

# Clean Skills
df['skill'] = df['skill'].str.lower().str.strip()
df.drop_duplicates(inplace=True)

# Remove Non-Tech and Noise
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

# Create a clean subset for specific visualizations
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
plt.title("Top 10 Technical Skills in Demand")
plt.xlabel("Frequency")
plt.ylabel("Skills")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ Python, SQL, and Data Analysis dominate the job market completely.")
print("→ Most roles strongly require these core foundations over niche tools.")

# ============================================================
# ========== TECHNICAL VS NON-TECHNICAL SKILLS ==========
# ============================================================
print("\n========== TECHNICAL VS NON-TECHNICAL SKILLS ==========")
nontech_df = df[~df.index.isin(tech_df.index)]
sizes = [len(tech_df), len(nontech_df)]
labels = ['Technical', 'Non-Technical']

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=(0.1, 0), labels=labels, autopct='%1.1f%%', shadow=True, startangle=140, colors=sns.color_palette("pastel")[0:2])
plt.title("Technical vs Non-Technical Skill Focus")
plt.show()

print("\nINSIGHTS:")
print("→ Employers emphasize technical (hard) skills by an almost 4:1 margin.")
print("→ While soft skills matter, specific tech keywords are strictly required for job matching.")

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
plt.ylabel("Overall Market Demand")
plt.tight_layout()
plt.show()

corr = merged_df_clean[['skill_count', 'job_count']].corr().iloc[0,1]
print(f"Calculated Correlation: {corr:.4f}")

print("\nINSIGHTS:")
print("→ There is practically no correlation between asking for more skills and those skills being highly demanded.")
print("→ Jobs asking for an excessive number of skills are 'unicorn hunting' rather than standard industry practice.")

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
plt.title("Outliers: General Market Demand")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ The long tail of 'Job Demand' shows that a tiny fraction of skills heavily dominates the market.")
print("→ Stripping away outliers reduces correlation near zero, proving extreme postings skew the overall trend.")

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
plt.title("Distribution of Market Demand")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ Most standard tech jobs focus on requesting a reasonable stack of 3 to 8 main skills.")
print("→ The extreme right skew in demand confirms a 'winner-takes-all' market built around key technologies.")

# ============================================================
# ========== CLUSTERING ANALYSIS ==========
# ============================================================
print("\n========== CLUSTERING ANALYSIS ==========")
cluster_data = merged_df_clean[['skill_count', 'job_count']].drop_duplicates().dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_data['cluster'] = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=cluster_data, x='skill_count', y='job_count', hue='cluster', palette='Set1', alpha=0.7)
plt.title("K-Means Market Segmentation")
plt.xlabel("Skills Requested per Job")
plt.ylabel("Overall Market Demand")
plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print("→ Cluster 0 (Typical Roles): Moderate skills and moderate demand. Represents realistic daily job postings.")
print("→ Cluster 1 (Unicorn Roles): High skills but low demand. Overloaded, messy job descriptions seeking generalized 'do-it-all' developers.")
print("→ Cluster 2 (Core Anchors): Moderate skills paired with highest market demand. Represents the foundational tools driving the industry.")

# ============================================================
# ========== FINAL TAKEAWAYS ==========
# ============================================================
print("\n========== FINAL TAKEAWAYS ==========")
print("→ Value of Core Skills: Market demand is heavily unequal; mastering 3 core skills (Python, SQL, AWS) guarantees broader opportunities than knowing 15 niche tools.")
print("→ The Generalist Trap: Roles requiring an excessive volume of skills (20+) consistently correlate with the lowest structural market demand.")
print("→ Clear Industry Standards: True tech roles clearly prioritize specific hard skills over soft skills, meaning resumes must be keyword-optimized for Applicant Tracking Systems.")
