import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
from scipy.stats import zscore

# Connect to database
conn = sqlite3.connect('Fantasy/yahoo-nhl-37045.db')

stats_info = pd.read_sql_query("SELECT rowid, abbr, name FROM stat", conn)

# draft pick stats
team_stats_query = """
SELECT mt.matchup_id, mt.team_id, ps.stat_id, ps.date_, w.idx as week_id,
    SUM(ps.value) AS total_stat_value,
    COUNT(ps.player_id) AS active_players
FROM player_stat  AS ps 
JOIN player AS p ON ps.player_id = p.rowid
JOIN week AS w ON ps.date_ BETWEEN w.start_ AND w.end_
JOIN matchup AS m ON m.week_id = w.idx
JOIN matchup_team AS mt ON mt.matchup_id = m.rowid
JOIN draft_pick AS dp ON dp.player_id = p.rowid AND dp.team_id = mt.team_id
GROUP BY mt.matchup_id, mt.team_id, ps.stat_id, ps.date_, w.idx
ORDER BY mt.matchup_id, mt.team_id, ps.stat_id
"""

team_stats = pd.read_sql_query(team_stats_query, conn)
print(f"Loaded {len(team_stats)} stat records")

if len(team_stats) == 0:
    print("No data found")
    conn.close()
    exit()

# matchup stats
teams = pd.read_sql_query("SELECT rowid, yahoo_key, name FROM team", conn)
teams['name'] = teams['name'].astype(str).str.replace("b'", "").str.replace("'", "")

matchup_results = pd.read_sql_query("""
    SELECT mt.matchup_id, mt.team_id, mt.yahoo_points, mt.is_winner, mt.is_tied, m.week_id
    FROM matchup_team AS mt 
    JOIN matchup AS m ON m.rowid = mt.matchup_id
""", conn)

print(f"Loaded {len(matchup_results)} matchup results")

conn.close()

# 1. Team stats by matchup

# Aggregate teams per matchup
team_matchup_stats = (
    team_stats.groupby(['matchup_id', 'team_id', 'stat_id'])['total_stat_value'].sum().reset_index())

# Merge with matchup results
team_matchup_stats = team_matchup_stats.merge(
    matchup_results[['matchup_id', 'team_id', 'yahoo_points', 'is_winner', 'week_id']], 
    on=['matchup_id', 'team_id'])

print(f"Team matchup stats: {len(team_matchup_stats)} records")

# 2. Calculate league averages

# weekly average
weekly_averages = (
    team_matchup_stats.groupby(['week_id', 'stat_id'])['total_stat_value']
    .agg(['mean', 'std', 'median'])
    .reset_index()
    .rename(columns={'mean': 'league_avg', 'std': 'league_std', 'median': 'league_median'}))

# standardize scores
team_matchup_stats = team_matchup_stats.merge(
    weekly_averages, on=['week_id', 'stat_id'])

# calc z-scores
team_matchup_stats['z_score'] = (
    (team_matchup_stats['total_stat_value'] - team_matchup_stats['league_avg']) / 
    team_matchup_stats['league_std']).fillna(0)

# 3. weightings

stat_weights = {
    # Skater categories 
    1: 1.0,   # Goals 
    2: 1.0,   # Assists 
    5: 1.0,   # Power Play Points 
    6: 1.0,   # Hits 
    8: 1.0,   # Blocks 
    
    # Goalie categories
    10: 1.0,  # Goalie Games Won 
    11: 1.0,  # Save Percentage 
    12: 1.0,  # Shutouts 
    
}
team_matchup_stats['stat_weight'] = team_matchup_stats['stat_id'].map(stat_weights).fillna(1.0)

save_pct_stat_id = 11 
team_matchup_stats.loc[team_matchup_stats['stat_id'] == save_pct_stat_id, 'z_score'] = (
    team_matchup_stats.loc[team_matchup_stats['stat_id'] == save_pct_stat_id, 'z_score']
)

# weighted z-score
team_matchup_stats['weighted_z_score'] = team_matchup_stats['z_score'] * team_matchup_stats['stat_weight']

# 4. expected performance

# weighted performance per team per matchup
team_expected_performance=(team_matchup_stats.groupby(['matchup_id', 'team_id', 'week_id']).agg({
        'weighted_z_score': 'sum',
        'z_score': 'mean',
        'yahoo_points': 'first',
        'is_winner': 'first'
    })
    .reset_index().rename(columns={'weighted_z_score': 'expected_performance_score'}))

# 5. strenght of schedule

team_strength = (team_expected_performance.groupby('team_id')['expected_performance_score'].mean().reset_index()
                 .rename(columns={'expected_performance_score': 'team_strength'})
)

# matchup strength
matchup_opponents = (team_expected_performance[['matchup_id', 'team_id', 'expected_performance_score']]
                     .rename(columns={'team_id': 'opponent_id', 'expected_performance_score': 'opponent_strength'})
)

matchup_pairings = (matchup_results.groupby('matchup_id')['team_id'].apply(list).reset_index())

# opponent mapping
opponent_map = []
for _, row in matchup_pairings.iterrows():
    matchup_id = row['matchup_id']
    teams_in_matchup = row['team_id']
    if len(teams_in_matchup) == 2:  # Head-to-head matchup
        opponent_map.append({
            'matchup_id': matchup_id,
            'team_id': teams_in_matchup[0],
            'opponent_id': teams_in_matchup[1]
        })
        opponent_map.append({
            'matchup_id': matchup_id,
            'team_id': teams_in_matchup[1],
            'opponent_id': teams_in_matchup[0]
        })

opponent_df = pd.DataFrame(opponent_map)
if len(opponent_df) > 0:
    # Merge opponent strength
    opponent_df = opponent_df.merge(team_strength, left_on='opponent_id', right_on='team_id', suffixes=('', '_opp'))
    opponent_df = opponent_df[['matchup_id', 'team_id', 'team_strength']].rename(columns={'team_strength': 'opponent_strength'})
    
    team_expected_performance = team_expected_performance.merge(
        opponent_df, on=['matchup_id', 'team_id'], how='left'
    )
    team_expected_performance['opponent_strength'] = team_expected_performance['opponent_strength'].fillna(0)
else:
    team_expected_performance['opponent_strength'] = 0

# 6. luck metrics
'''
luck_per_team[['name', 'luck_amount']]
luck_per_team.head()
'''
# Performance vs outcome
team_expected_performance['performance_luck'] = np.where(team_expected_performance['is_winner'] == 1,
                                                         -team_expected_performance['expected_performance_score'], 
                                                         team_expected_performance['expected_performance_score'])

# Strength-adjusted luck 
# TODO might need to adjust the dampening factor
team_expected_performance['strength_adjusted_luck'] = (team_expected_performance['performance_luck'] + 
                                                       team_expected_performance['opponent_strength'] * 0.3)

# expected win probablity 
def performance_win_prob(performance_diff):
    return 1 / (1 + np.exp(-performance_diff))

# performance diff per opponent
performance_diffs = []
for _, row in team_expected_performance.iterrows():
    matchup_id = row['matchup_id']
    team_perf = row['expected_performance_score']
    
    # Find opponent performance in same matchup
    opponent_perf = team_expected_performance[
        (team_expected_performance['matchup_id'] == matchup_id) & 
        (team_expected_performance['team_id'] != row['team_id'])
    ]['expected_performance_score']
    
    if len(opponent_perf) > 0:
        perf_diff = team_perf - opponent_perf.iloc[0]
    else:
        perf_diff =0
    
    performance_diffs.append(perf_diff)

team_expected_performance['performance_differential'] = performance_diffs
team_expected_performance['expected_win_prob'] = performance_win_prob(team_expected_performance['performance_differential'])

# Expected vs actual wins
team_expected_performance['win_probability_luck'] = (
    team_expected_performance['is_winner'] - team_expected_performance['expected_win_prob']
)

# 7. total luck per team

team_luck_summary = (team_expected_performance.groupby('team_id').agg({
        'performance_luck': ['sum', 'mean'],
        'strength_adjusted_luck': ['sum', 'mean'], 
        'win_probability_luck': ['sum', 'mean'],
        'expected_win_prob': 'sum',
        'is_winner': 'sum',
        'matchup_id': 'count'
    })
    .round(3)
)

team_luck_summary.columns = [f"{col[0]}_{col[1]}" for col in team_luck_summary.columns]
team_luck_summary = team_luck_summary.reset_index()

# comp luck score
#can change the factors here too
team_luck_summary['composite_luck_score'] = (
    team_luck_summary['performance_luck_sum'] * 0.3 +
    team_luck_summary['strength_adjusted_luck_sum'] * 0.4 +
    team_luck_summary['win_probability_luck_sum'] * 0.3
)

team_luck_summary = team_luck_summary.merge(teams[['rowid', 'name']], left_on='team_id', right_on='rowid')
team_luck_summary = team_luck_summary.sort_values('composite_luck_score', ascending=False)

display_cols = [
    'name', 
    'composite_luck_score',
    'is_winner_sum',
    'expected_win_prob_sum', 
    'performance_luck_sum',
    'strength_adjusted_luck_sum',
    'win_probability_luck_sum',
    'matchup_id_count'
]

display_df = team_luck_summary[display_cols].copy()
display_df.columns = [
    'Team', 
    'Composite Luck Score',
    'Actual Wins',
    'Expected Wins', 
    'Performance Luck',
    'Strength-Adj Luck',
    'Win Probability Luck',
    'Total Matchups'
]

print(display_df.to_string(index=False))

# Visualizations

num_teams = len(team_luck_summary)
colors = plt.cm.tab10(np.linspace(0, 1, num_teams))

# Composite Luck Score
plt.figure(figsize=(12, 8))
bars1 = plt.bar(range(len(team_luck_summary)), team_luck_summary['composite_luck_score'], color=colors)
plt.xticks(range(len(team_luck_summary)), team_luck_summary['name'], rotation=45, ha='right')
plt.title('Composite Luck Score', fontsize=16, fontweight='bold')
plt.ylabel('Luck Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
             f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Expected vs actual
plt.figure(figsize=(10, 8))
scatter = plt.scatter(team_luck_summary['expected_win_prob_sum'], 
                     team_luck_summary['is_winner_sum'], 
                     c=colors, s=150, alpha= 0.7, edgecolors='black', linewidth=1)
plt.plot([0, team_luck_summary['expected_win_prob_sum'].max()], 
         [0, team_luck_summary['expected_win_prob_sum'].max()], 'r--', alpha=0.5, linewidth=2)

# steps
max_wins = max(team_luck_summary['expected_win_prob_sum'].max(), team_luck_summary['is_winner_sum'].max())
tick_step = 1.0 
ticks = np.arange(0, max_wins + tick_step, tick_step)

plt.xticks(ticks)
plt.yticks(ticks)

plt.xlabel('Expected Wins', fontsize=12)
plt.ylabel('Actual Wins', fontsize=12)
plt.title('Expected vs Actual Wins', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add team labels to lessen the wonk
for i, row in team_luck_summary.iterrows():
    plt.annotate(row['name'], 
                (row['expected_win_prob_sum'], row['is_winner_sum']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, alpha=0.8, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    

plt.tight_layout()
plt.show()

# luck components seperated
plt.figure(figsize=(14, 8))
x = np.arange(len(team_luck_summary))
width = 0.25

bars1 = plt.bar(x - width, team_luck_summary['performance_luck_sum'], width, 
                label='Performance Luck', alpha=0.8, color='skyblue')
bars2 = plt.bar(x, team_luck_summary['strength_adjusted_luck_sum'], width, 
                label='Strength-Adjusted Luck', alpha=0.8, color='lightcoral')
bars3 = plt.bar(x + width, team_luck_summary['win_probability_luck_sum'], width, 
                label='Win Probability Luck', alpha=0.8, color='lightgreen')

plt.xlabel('Teams', fontsize=12)
plt.ylabel('Luck Score', fontsize=12)
plt.title('Luck Components by Team', fontsize=16, fontweight='bold')
plt.xticks(x, team_luck_summary['name'], rotation=45, ha='right')
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# performance vs luck(quadrants)
plt.figure(figsize=(12, 10))

performance_by_team = team_expected_performance.groupby('team_id')['expected_performance_score'].mean().reset_index()
performance_by_team.columns = ['team_id', 'avg_performance']

# merge team luck summary
plot_data = team_luck_summary.merge(performance_by_team, on='team_id', how='left')

scatter2 = plt.scatter(plot_data['avg_performance'], 
                      plot_data['composite_luck_score'], 
                      c=colors, s=150, alpha=0.7, edgecolors='black', linewidth=1)

plt.xlabel('Average Performance Score', fontsize=12)
plt.ylabel('Composite Luck Score', fontsize=12)
plt.title('Performance vs Luck Analysis', fontsize=16, fontweight='bold')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
plt.grid(True, alpha=0.3)

x_margin = (plot_data['avg_performance'].max() - plot_data['avg_performance'].min()) * 0.15
y_margin = (plot_data['composite_luck_score'].max() - plot_data['composite_luck_score'].min()) * 0.15
plt.xlim(plot_data['avg_performance'].min() - x_margin, plot_data['avg_performance'].max() + x_margin)
plt.ylim(plot_data['composite_luck_score'].min() - y_margin, 
         plot_data['composite_luck_score'].max() + y_margin)

xlims = plt.xlim()
ylims = plt.ylim()

x_offset = (xlims[1] - xlims[0]) * 0.05
y_offset = (ylims[1] - ylims[0]) * 0.05

# good and lucky
plt.text(xlims[1] - x_offset, ylims[1] - y_offset, 
         'SKILLED &\nLUCKY', 
         fontsize=14, fontweight='bold', alpha= 0.8,
         ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4, edgecolor='darkgreen'))

# bad and lucky 
plt.text(xlims[0] + x_offset, ylims[1] - y_offset, 
         'LUCKY BUT\nUNDERPERFORMING', 
         fontsize=14, fontweight='bold', alpha= 0.8,
         ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.4, edgecolor='orange'))

# good and unlucky
plt.text(xlims[1] - x_offset, ylims[0] + y_offset, 
         'SKILLED BUT\nUNLUCKY', 
         fontsize=14, fontweight='bold', alpha= 0.8,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.4, edgecolor='darkred'))

# bad and unlucky
plt.text(xlims[0] + x_offset, ylims[0] + y_offset, 
         'POOR PERFORMANCE\n& UNLUCKY', 
         fontsize=14, fontweight='bold', alpha= 0.8,
         ha='left', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.4, edgecolor='gray'))

for idx, row in plot_data.iterrows():
    perf_score = row['avg_performance']
    luck_score = row['composite_luck_score']
    team_name = row['name']
    
    if pd.notna(perf_score) and pd.notna(luck_score):
        plt.annotate(team_name, 
                    (perf_score, luck_score),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, alpha= 0.9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()
