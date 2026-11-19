# GTA VI Trophy Rarity Forecast

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ridwanter0319/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ridwanter0319/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun) bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI PS5 trophy, its Exophase earn rate
at 2026-11-26T00:00:00Z as five quantiles (5/25/50/75/95%), scored by mean
pinball loss on the logit scale.

## Model

`main.py` (Python 3, standard library only) reads one case file and:

- collapses the read history to distinct page states, since Exophase
  refreshes are step changes and duplicate reads would flatten any fit;
- models each trophy's rate on the logit scale against log age since
  release (the shape of a lognormal time-to-earn curve), blending a
  recent-window and a full-history weighted least squares fit;
- shrinks each trophy's slope towards its trophy-class prior (story,
  mission, activity, challenge, grind, completion, platinum), with
  shrinkage driven by the earner count behind the last rate;
- handles barely-refreshed histories as a dilution regime: the next
  refresh is expected to multiply the tracked pool with less progressed
  players, so the median mixes a diluted state with a grown state under
  a wide spread;
- builds quantiles from a per-trophy sigma (horizon, fit residuals,
  slope error over the extrapolation gap, counting noise), then enforces
  story-chain monotonicity and caps the platinum at the rarest other
  trophy, since its earners are a subset of every other trophy's.

The program is deterministic and runs in well under the 8 second cap.
