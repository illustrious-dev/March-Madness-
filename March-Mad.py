import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Real-Life NCAA Tournament Data
teams = pd.read_csv('ncaa_teams.csv')  # Replace with actual dataset path
betting_odds = pd.read_csv('ncaa_betting_odds.csv')
injury_reports = pd.read_csv('ncaa_injuries.csv')

# Feature Engineering
def calculate_fatigue(team):
    """ Estimate fatigue based on recent game schedule."""
    return team['games_last_10_days'] / 10

def identify_cinderella_teams(team):
    """ Identify Cinderella teams based on offensive and defensive performance."""
    return (
        (team['ppg'] > 75) &  # High scoring offense
        (team['opp_ppg'] < 65) &  # Strong defense
        (team['three_pt_pct'] > 0.35) &  # Good 3PT shooting
        (team['upper_classmen_exp'] > 2)  # Experienced team
    )

teams['fatigue'] = teams.apply(calculate_fatigue, axis=1)
teams['cinderella'] = teams.apply(identify_cinderella_teams, axis=1)

# Merge Data
full_data = teams.merge(betting_odds, on='team_id').merge(injury_reports, on='team_id')

# Prepare Training Data
X = full_data[['ppg', 'opp_ppg', 'three_pt_pct', 'upper_classmen_exp', 'fatigue', 'betting_odds']]
y = full_data['tournament_success']  # Binary label (win/loss)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Simulate Tournament
def simulate_bracket(teams, model):
    """ Predict each game outcome and generate a tournament bracket."""
    bracket = []
    round_num = 1
    while len(teams) > 1:
        print(f'Round {round_num}:')
        winners = []
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):  # Handle odd number of teams
                winners.append(teams.iloc[i])
                continue
            game_data = teams.iloc[[i, i+1]][X.columns]
            prediction = model.predict(game_data)
            winner = teams.iloc[i if prediction[0] == 1 else i+1]
            winners.append(winner)
            print(f'{teams.iloc[i]["team_name"]} vs {teams.iloc[i+1]["team_name"]} -> Winner: {winner["team_name"]}')
        teams = pd.DataFrame(winners).reset_index(drop=True)
        bracket.append(teams)
        round_num += 1
    print(f'Champion: {teams.iloc[0]["team_name"]}')
    return bracket

bracket = simulate_bracket(teams, model)
