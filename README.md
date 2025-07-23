# Fantasy Hockey Luck Analysis 🏒📊

A comprehensive statistical analysis where I pit my friends against each other as we analyze which teams lucked their way to the top, and which ones deserved their placements.

Credit to [j-krl](https://github.com/j-krl/yfpy-nhl-sqlite?tab=readme-ov-file) for providing the schema to help extract the data from the league.

## 🎯 Project Overview

This project analyzes fantasy hockey performance to determine which teams are genuinely skilled versus the lucky ones amongst us. Using statistical modeling, strength-of-schedule adjustments, and win probability calculations, this analysis can cause feuds that could last multiple seasons.

Helps provide insight on factors beyond skill and strategic decision making, displaying those that may be over/under-performing.

## 📈 Key Features

- **Multi-dimensional Luck Quantification**: Combines performance metrics, opponent strength, and win probability
- **Statistical Modeling**: Z-score normalization, weighted performance scoring, and logistic regression
- **Comprehensive Visualizations**: Quadrant analysis, trend charts, and component breakdowns

## 🛠️ Tech Used

- **Python**: Data processing and statistical analysis
- **SQLite**: Database management and querying
- **Pandas/NumPy**: Data manipulation and numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **SciPy**: Advanced statistical functions

## 📊 Results

![performancevsLuck](https://github.com/user-attachments/assets/df46ab26-8207-48cf-8fb6-b1ec42839ade)
Plotting average performance against composite luck score. 
- Upper Right(Skilled and Lucky): Strong teams that are likely contenders
- Upper Left(Lucky and Underperforming): Unsustainable pace that is likely to regress
- Bottom Right(Skilled and Unlucky): Strong teams that are due for some positive regression
- Lower Left(Poor and Unlucky): Struggling teams that just can't find their footing
![Luck Components By Team](https://github.com/user-attachments/assets/1e4a9bd5-b4dd-4ecf-b081-2299365884cf)
Grouped bar chart that splits into 3 luck components: Performance luck, strength-of-schedule, and win probability. This helps provide an understanding of where the team's luck is coming from.
![ExpectedVsActual](https://github.com/user-attachments/assets/1b9d0fb7-0886-466f-b63a-eba633d9c82d)
Compares predicted wins against actual wins, showing which teams outperformed/underperformed expectations.
![compositeLuckScore](https://github.com/user-attachments/assets/6ac362c5-33e6-4fbc-ae4c-63178ca5f418)
Ranked bar chart displaying each team's overall luck score as one metric. Weighted composite luck score (30% performance luck + 40% strength-adjusted luck + 30% win probablity)
