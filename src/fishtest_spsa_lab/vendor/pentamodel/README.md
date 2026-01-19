# PentaModel

PentaModel simulates chess match outcomes using a logistic function to estimate win, loss, and draw probabilities based on evaluation scores and Elo difference.

## Features

- Calculates win/loss/draw probabilities for given evaluation and Elo difference
- Computes expected scores and Elo differences
- Simulates game pairs and pentanomial match outcomes
- Provides probability distributions for all possible results

## Usage

```python
from pentamodel import PentaModel

model = PentaModel(opponentElo=100)
print(model.gamePairProbs)        # Probabilities for all game pair outcomes
print(model.pentanomialProbs)     # Probabilities for pentanomial outcomes
print(model.return_game_pair())   # Randomly sampled game pair outcome
print(model.return_match_pentanomial(1000))  # Simulate 1000 game pairs
