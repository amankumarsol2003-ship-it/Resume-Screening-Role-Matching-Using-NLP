# AI-Powered Resume Screening & Role Matching System (NLP)

An enterprise-grade talent acquisition pipeline that automates resume screening and candidate role-matching using advanced Natural Language Processing (NLP) and vector space modeling. The system extracts structural characteristics and semantic text features from unstructured resume summaries, scores candidate compatibility against custom-defined organizational role profiles, and ranks applicants using high-precision evaluation matrices.

## 🚀 Key Project Achievements & Results
* **High-Precision Ranking (NDCG@10):** Achieved an ideal normalized discounted cumulative gain score of **1.0 (100%)** on targeted evaluation roles, indicating optimal algorithmic candidate ranking performance.
* **Large-Scale Processing:** Cleaned, processed, and evaluated a dense corpus of **5,000 distinct candidate records** across 34 engineering, domain, and experience variables.
* **Engineered Composite Scoring Engine:** Implemented a multi-tier matching formula combining Global Semantic Text Similarity (50%), Exact-Match Skill Coverage (30%), and Experience Fit thresholds (20%) to prevent generic keyword stuffing over true technical tenure.

## 🛠️ Tech Stack, Frameworks & Libraries
* **Core Language:** Python
* **Natural Language Processing:** Scikit-learn (`TfidfVectorizer`, sublinear term frequency scaling, customized token boundary filters)
* **Mathematical & Vector Similarity operations:** Scikit-learn (`cosine_similarity`), NumPy
* **Evaluation & Ranking Operations:** Scikit-learn (`ndcg_score`, `precision_score`, `recall_score`, `f1_score`)
* **Data Pipelines & Visualizations:** Pandas, Seaborn, Matplotlib

## 📊 Core Data Architecture
The operational pipeline structures and transforms data utilizing three core system files:
1. `parsed_resumes_cleaned.csv`: The clean master data file containing 5,000 rows across 34 technical, operational, and managerial features.
2. `resume_role_composite_matching_results.csv`: System output detailing calculated text similarity, categorical skill coverage percentages, experience fit, and final weighted composite scores.
3. `resume_matching_evaluation_metrics.csv`: Stores historical validation scores including ranking validation parameters (Precision@K, Recall@K, and NDCG).

### Dataset Feature Blueprint (Sample of 34 System Variables)
* **Metadata Targets:** `candidate_id`, `resume_file_name`, `primary_domain`, `primary_role`
* **Numerical Metrics:** `years_experience`, `resume_length_words`, `keyword_density_score`, `profile_completeness_score`
* **Operational Tenure Flags:** `management_experience_flag`, `project_management_experience_flag`, `agile_scrum_experience_flag`, `client_facing_experience_flag`, `ml_experience_flag`, `cloud_experience_flag`
* **Unstructured Text Buffers:** `technical_skills_raw`, `tools_platforms_raw`, `experience_summary`, `project_summary`, `key_achievements`

## ⚙️ Algorithmic Workflow & Pipeline Execution

### 1. Robust NLP Text Cleaning & Tokenization
* Custom multi-regex sanitization layer strip URLs, system metadata, and emails while explicitly preserving syntax-critical technical designations (e.g., `C++`, `C#`, `.net`).
* Content matrices merge disparate skill arrays, execution summaries, and achievement text fields into unified holistic string vectors per applicant.

### 2. Semantic Text Vectorization (TF-IDF Space)
* Text indices are vectorized into high-dimensional space using sublinear term-frequency scaling to compress the impact of repetitive language.
* Unigram and Bigram combinations (`ngram_range=(1, 2)`) extract relational industry phrases rather than isolated keywords, capped at a dense top-feature boundary (`max_features=10000`).

### 3. Weighted Composite Scoring Engine
Candidates are scored against targeted profiles (e.g., Data Scientist, Data Analyst, Software Developer) utilizing an elastic multi-component weight matrix:
\[\text{Composite Score} = (0.50 \times \text{Text Similarity}) + (0.30 \times \text{Skill Coverage}) + (0.20 \times \text{Experience Fit})\]
* **Text Similarity:** Normalized Cosine Similarity calculation between candidate TF-IDF vectors and the core target baseline profile matrix.
* **Skill Coverage:** Phrase-boundary boundary exact matching (`(?<!\w)skill(?!\w)`) testing for mandated foundational stack items to avoid out-of-context phrase matches.
* **Experience Fit:** Non-linear scaling index tracking applicant tenure against role baseline minimums: \(\min(\text{Tenure} / \text{Minimum Required}, 1.0)\).

### 4. Ranking Evaluation & Performance Analytics
* Generates validation labels across composite score margins using threshold segmentation techniques.
* Computes positional ranking quality metrics (**NDCG@K**) to ensure top-tier algorithmic selections mirror qualitative recruitment preferences.

## 💻 Local Quickstart & Replication

1. **Clone this repository and verify system directories:**
   ```bash
   git clone https://github.com
   cd Resume-Screening-Project
   ```

2. **Install core environment dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the text preprocessing and matching scripts:**
   ```bash
   jupyter notebook Resume_Screening_and_Role_matching_Capstone_Project.ipynb
   ```
