## Abstract

Depression, anxiety, and obesity have been on the rise in the United States for decades. Simulatneously, religion has been on the decline. This project will look to determine whether this is coincidence or correlation by exploring faith, success, and happiness in the United States. Using Pew Research data with over 130 different features and 37,000 respondents from all 50 states in 2023 and 2024, the analysis will include religious, demographic, political, and other features in relation to targets of happiness, income, and educational attainment. The hypothesis is that having a strong sense of religion and practicing regularly will benefit your happiness levels, although it may not necessarily benefit your income and educational attainment levels since religious people tend to center God in their lives, whereas nonreligious people center career in theirs.

This project will use both Statisical Modeling and Machine Learning methodologies. For statistical modeling, it will use Chi-Square testing since the data is entirely categorical. Milestone 1 will solely look at the RELPER (extent of religion) and CURREL (specific religious affiliation) features as inputs. It will also look at the HAPPY (self-rated happiness level) and SUCCESS (sum of HAPPY, INC_SDT1 for income level, and EDUCREC for educational attainment level) features as the targets. Milestone 2 may expolore other features and targets, such as immigration status, internet usage, number of kids, political views, and more. For the machine learning part of Milestone 1, Logistic Regression is used. Due to the unbalanced nature of the data, balancing parameters were used to help prevent overfitting and only one outcome prediction. A combination of RELPER and CURREL were also used to see if the interaction yielded a better model. In the future, more model parameters and interactions will be explored to improve the model. More ML classification methodologies will also be explored as the semester progresses, such as random forests, PCA with Trees, and Neural Networks.

The data comes from Pew Research Center: https://www.pewresearch.org/dataset/2023-24-religious-landscape-study-rls-dataset/


## Files
The analysis.ipynb contains all of the statistical and ML analysis done throughout Milestones 1 and 2
The app folder contains the webapp.py script for running the webapp. It also contains files where I attempted to
host the CSV on Google Cloud since it's too large to push to the repo. The webapp script still refers to the local
file however, as the Google Cloud Services was disabled for Lehigh Accounts (need admin approval for billing).
I also was not allowed to install the Git large file upload package on my Merck laptop due to admin requirements.


## Pacakges (also listed in requirements.txt)

Packages required:
- pandas
- matplotlib
- numpy
- seaborn
- scipy
- sklearn
- itertools
- streamlit

## Running the Web App

Run the following command after you've added the CSV data file to the data folder:

```bash
streamlit run webapp.py
```


## Setup (THIS DOESN'T WORK SINCE LEHIGH GOOGLE CLOUD SERVICES BILLING IS DISABLED)

1. Clone the repository.
2. Run the following commands:

```bash
python app/get_data.py && python app/webapp.py
```
