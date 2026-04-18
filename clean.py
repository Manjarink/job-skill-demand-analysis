import pandas as pd

# Load data
skills_df = pd.read_csv("job_skills.csv", nrows=150000)
jobs_df = pd.read_csv("linkedin_job_postings.csv", nrows=150000)

# Check columns
print(skills_df.columns)
print(jobs_df.columns)

# Clean data
skills_df.dropna(inplace=True)
jobs_df.dropna(inplace=True)

skills_df.drop_duplicates(inplace=True)
jobs_df.drop_duplicates(inplace=True)

# Rename for consistency
skills_df.rename(columns={'job_link': 'job_id'}, inplace=True)
jobs_df.rename(columns={'job_link': 'job_id'}, inplace=True)

# Standardize skills text
skills_df['job_skills'] = skills_df['job_skills'].str.lower().str.strip()

# SPLIT skills into list
skills_df['job_skills'] = skills_df['job_skills'].str.split(',')

# EXPLODE (VERY IMPORTANT)
skills_df = skills_df.explode('job_skills')

# Clean each skill
skills_df['job_skills'] = skills_df['job_skills'].str.strip()

# Rename column to simple name
skills_df.rename(columns={'job_skills': 'skill'}, inplace=True)

# Merge datasets
df = pd.merge(skills_df, jobs_df, on="job_id")

# Final check
print(df.head())

#MODIFY CLEANING FILE
df.to_csv("cleaned_data.csv", index=False)
