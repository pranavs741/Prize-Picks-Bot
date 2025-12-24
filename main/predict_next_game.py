"""
Script to predict Stephen Curry's points for the next game
Next game: Orlando Magic at Home (Today)
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

# Model definition (same as notebook)
class EnhancedPointsLSTM(nn.Module):
    def __init__(self, n_future_days, input_size=4, hidden_size=16, num_layers=1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, n_future_days)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]
        last = self.dropout(last)
        output = self.fc(last)
        return output

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load data
print("\nLoading data...")
curry_game_logs = pd.read_csv("curry_game_logs.csv")
opponent_matchup = pd.read_csv("opponent_matchup.csv")
teammate_availability = pd.read_csv("teammate_availability.csv")

# Convert dates to datetime
curry_game_logs['game_date'] = pd.to_datetime(curry_game_logs['game_date'])
opponent_matchup['game_date'] = pd.to_datetime(opponent_matchup['game_date'])
teammate_availability['game_date'] = pd.to_datetime(teammate_availability['game_date'])

# Merge dataframes
merged_data = curry_game_logs.merge(
    opponent_matchup, 
    on=['game_date', 'season'], 
    how='left', 
    suffixes=('', '_opp')
).merge(
    teammate_availability,
    on=['game_date', 'season'],
    how='left'
)

# Sort by date
merged_data = merged_data.sort_values('game_date').reset_index(drop=True)

# Feature columns
feature_cols = ['pts', 'min', 'fgm', 'fg3m']
n_previous_days = 10
n_future_days = 1

# Fill NaN values
for col in feature_cols:
    if col in merged_data.columns:
        merged_data[col] = merged_data[col].fillna(0)
    else:
        merged_data[col] = 0

# Extract feature data
feature_data = merged_data[feature_cols].values.astype(np.float32)

# Normalize features
scaler = StandardScaler()
feature_data_scaled = scaler.fit_transform(feature_data)

# Load trained model
print("Loading trained model...")
n_features = len(feature_cols)
model = EnhancedPointsLSTM(n_future_days=n_future_days, input_size=n_features).to(device)
model.load_state_dict(torch.load('best_model.pth', weights_only=True, map_location=device))
model.eval()

# Get the last n_previous_days games
print("\nAnalyzing last 10 games...")
last_games = merged_data[feature_cols].tail(n_previous_days).values.astype(np.float32)

# Show recent games info
recent_games = merged_data.tail(n_previous_days)[['game_date', 'opponent', 'home_away', 'pts']]
print("\nLast 10 games:")
print(recent_games.to_string(index=False))

# Scale using the same scaler
last_games_scaled = scaler.transform(last_games)

# Convert to tensor and add batch dimension
input_tensor = torch.tensor(last_games_scaled).unsqueeze(0).float().to(device)

# Make prediction
print("\nMaking prediction...")
with torch.no_grad():
    prediction_scaled = model(input_tensor).cpu().numpy()[0, 0]

# Inverse transform to get actual points
dummy = np.zeros((1, len(feature_cols)))
dummy[0, 0] = prediction_scaled
predicted_points = scaler.inverse_transform(dummy)[0, 0]

# Calculate statistics
recent_points = merged_data['pts'].tail(10).values
avg_recent = recent_points.mean()
std_recent = recent_points.std()

# Show results
print("\n" + "=" * 60)
print("STEPHEN CURRY POINTS PREDICTION")
print("=" * 60)
print(f"\nNext Game: Orlando Magic @ Golden State Warriors (Home)")
print(f"Game Date: Today")
print(f"\n{'='*60}")
print(f"PREDICTED POINTS: {predicted_points:.1f}")
print(f"{'='*60}")
print(f"\nRecent Performance (Last 10 games):")
print(f"  Average: {avg_recent:.1f} points")
print(f"  Standard Deviation: {std_recent:.1f} points")
print(f"  Range: {recent_points.min():.0f} - {recent_points.max():.0f} points")
print(f"\nRecent game-by-game points: {recent_points}")
print(f"\nNote: This prediction is based on Curry's recent performance")
print(f"      patterns. Actual performance may vary due to game flow,")
print(f"      opponent defense, and other factors.")




