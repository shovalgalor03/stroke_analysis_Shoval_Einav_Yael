# Stroke Analysis Project 

## Project Overview
This project performs a comprehensive statistical and data analysis on a healthcare dataset to identify risk factors associated with stroke. The analysis pipeline includes data cleaning, feature engineering, statistical testing (Chi-Square, Relative Risk), and unsupervised machine learning (K-Means Clustering, PCA) to profile patient risk groups.

**Authors:** Shoval Galor, Einav Tzemach, Yael Hezkiyahu

## Data Description
The project uses the "healthcare-dataset-stroke-data.csv" file. 
The dataset includes clinical and demographic information for each patient, used to predict or analyze the likelihood of a stroke.

**Key Attributes:**
* **Demographics:** "gender", "age", "ever_married", "Residence_type".
* **Medical History:** "hypertension", "heart_disease", "stroke" (The Target Variable).
* **Clinical Measurements:** "avg_glucose_level", "bmi".
* **Lifestyle:** "work_type", "smoking_status".

**Link to Dataset:**
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

   ## References
1.  **Dataset Source:** fedesoriano. (2020). *Stroke Prediction Dataset*. Retrieved from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
       Confidential Source (intended for educational purposes only).
2.  **Background Info:** World Health Organization (WHO) global stroke statistics.

## Instructions for Running the Project

# Prerequisites
* Python 3.8 or higher.
* The following Python libraries (see "requirements.txt"):
    * pandas
    * numpy
    * matplotlib
    * seaborn
    * scikit-learn
    * scipy
    * statsmodels

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/shovalgalor03/stroke_analysis_shoval_einav_yael.git](https://github.com/shovalgalor03/stroke_analysis_shoval_einav_yael.git)
    cd stroke_analysis_shoval_einav_yael
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
To run the full analysis pipeline (Loading -> Cleaning -> Statistics -> Clustering -> Visualization), execute the "main.py" script:

```bash
python main.py
