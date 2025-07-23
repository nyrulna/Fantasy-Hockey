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

## 🔍 Key Insights & Project Findings

### What the Data Revealed
After analyzing the entire season, several patterns emerged that go beyond just "who got lucky":

---

### **The Playoff Reality Check**
- **"Chill guy" (1st place):** Genuine skill with moderate positive luck – a well-deserved champion.
- **Teams 2–4:** Mixed skill vs. luck factors, showing the playoff race was more competitive than standings suggested.
- **Bottom teams:** Some faced both performance issues **and** bad luck (a double whammy).

---

### **Unexpected Statistical Discoveries**

#### **1. Strength of Schedule Mattered More Than Expected**
- Some teams’ records were inflated by facing weaker opponents during hot streaks.  
- Others struggled due to repeatedly facing strong teams during their peak weeks.  
- The **0.3 dampening factor** for opponent strength proved **critical for accuracy**.

#### **2. Win Probability vs. Reality**
- Several teams were winning close matchups at **unsustainable rates** (classic luck).  
- Others lost tight games despite strong performance.  
- The **logistic regression model** highlighted these patterns better than raw win/loss records.

#### **3. Composite Scoring Revealed Hidden Trends**
- Weighted approach (30% performance + 40% strength-adjusted + 30% win probability)  
  **exposed teams due for major regression.**  
- Some “good” teams were simply benefiting from **timing and matchups**,  
  while others were **undervalued threats.**

---
