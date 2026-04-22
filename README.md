# Analysis of Technical Skills Demand in Job Market using Exploratory Data Analysis (EDA)

**Data Science Minor Project (Jan-April 2026)**

## Introduction

Technical skills play a crucial role in determining employability. Organizations increasingly rely on data-driven hiring practices, where specific technical competencies are prioritized.

This project focuses on analyzing job market data to identify trends in skill demand. By applying Exploratory Data Analysis (EDA) techniques—including statistical correlations, distribution modeling, and K-Means clustering—insights are derived regarding market demand.

### Objectives
*   Identify the most frequently requested technical skills in the dataset.
*   Analyze the distribution of job demand and skill volume.
*   Evaluate the relationship between the number of skills and market demand.
*   Segment the tech job market dynamically using clustering algorithms.

## Dataset Details

The dataset was obtained from Kaggle and contains job postings and associated skill requirements collected from online job portals. 

Two primary datasets were integrated using a common identifier (`job_id`):
1.  **Job Postings Dataset**: Includes job title, company name, job location, job level, and type.
2.  **Skills Dataset**: Lists specific skills required for each job.

The unified dataset consists of over 100,000 records.

## Exploratory Data Analysis (EDA) Process

1.  **Data Cleaning**: Handled null values, removed duplicates, and standardized text (lowercase conversion, spacing adjustments).
2.  **Data Transformation**: Processed multiple skills per job posting by splitting them into individual components.
3.  **Data Filtering**: Isolated technical job roles using explicit keywords (e.g., data, developer, engineer, analyst, AI, machine learning).
4.  **Noise Removal**: Removed non-technical skills (e.g., communication, teamwork) and patterned noise (e.g., degree requirements).
5.  **Feature Engineering**: Derived quantitative features including `skill_count` (total listed skills per job) and `job_count` (frequency of specific skill demand).

## Key Analysis and Findings

### 1. Top Technical Skills vs. Non-Technical Priority
*   **Most Frequent Skills**: Python and SQL appear more frequently than any other technologies or skills.
*   **Skill Prioritization**: Mentions of technical skills outnumber non-technical soft skills by nearly a 4:1 ratio.

### 2. Skill Count vs. Job Demand
*   **Correlation**: A weak negative correlation (~ -0.06) combined with a slight inverse regression slope indicates no strong linear relationship.
*   **Insight**: Posting a larger volume of skills does not correspond to proportionally higher Job Demand values.

### 3. Outliers and Statistical Distribution
*   **Distribution**: Both variance and distribution analyses display a right-skew. A majority of highly specialized skills have low demand frequency, while a small subset of skills appears disproportionately often.
*   **Insight**: The data confirms a top-heavy distribution built around frequently requested technologies, with postings requesting 15+ skills existing as statistically infrequent outliers.

### 4. Market Segmentation (K-Means Clustering)
*   **Cluster Approach**: The data was segmented into tiers based on skill volume and demand using $k=3$ K-Means clustering.
*   **Results**:
    *   **Cluster 0**: Shows moderate skill ranges mapped to average demand.
    *   **Cluster 1**: Shows higher average skill counts but distinctly lower Job Demand trends.
    *   **Cluster 2**: Shows a group with relatively higher Job Demand compared to other clusters.

## Conclusion

The analysis indicates that job demand is concentrated around a smaller set of frequently occurring skills. Additionally, increasing the number of listed skills does not show a strong relationship with higher demand.

## Practical Implications

*   **Focus on high-frequency skills** such as Python and SQL.
*   **Avoid listing excessive unrelated skills** on resumes or job descriptions.
*   **Target a balanced skill set**, as the data indicates typical roles actively request 3–8 core skills.

## Future Scope

*   **Skill Gap Analyzer System**: An application where individuals input current skills to generate a personalized learning report based on data trends.
*   **Career Recommendation Engine**: Algorithmic suggestions for roles based on an individual's technical array.
*   **Real-Time Data Pipelines**: Integrating live job-board APIs for continuously updated statistics.
*   **Advanced NLP extraction**: Employing deep Natural Language Processing techniques to extract nuanced requirements directly from raw job descriptions.

## References
*   Kaggle Dataset: Job Skills and Job Postings Data
*   Pandas: https://pandas.pydata.org/
*   Matplotlib: https://matplotlib.org/
*   Seaborn: https://seaborn.pydata.org/
*   Scikit-learn: https://scikit-learn.org/
