"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model, in one paragraph: every rate is modelled on the logit scale against
log age (hours since release), which is the shape a lognormal time-to-earn
distribution gives a closed cohort of players.  Per trophy, a weighted
least squares line is fitted over the distinct page states in the history
(Exophase refreshes are step changes, so duplicate reads are collapsed
first), once over the recent window and once over the whole history, and
the two are blended.  Each trophy's slope is shrunk towards the mean slope
of its trophy class (story, mission, activity, challenge, grind,
completion, platinum), with the shrinkage strength driven by how many
earners back the trophy's last rate: a rate printed from four earners is
mostly prior, a rate printed from four thousand is mostly its own data.
The forecast is anchored on a smoothed last logit plus the shrunk slope
times the log-age gap to the target, softly capped by a per-class ceiling
and a dilution floor (earners already counted cannot un-earn a trophy;
only new tracked players can push a rate down, and only so far).

Histories with almost no refresh in them are a different regime: the page
is stale, the next refresh will mostly dilute rates with newly tracked,
less progressed players, so the median is built from a mixture of a
diluted state (new cohort arriving at a class-dependent relative rate)
and a grown state (the old cohort simply progressing), with a wide
spread that covers both.

Quantiles are the median plus the normal z of each level times a
per-trophy sigma on the logit scale, assembled from the horizon, the
fit's residual scatter, the slope's standard error over the extrapolation
gap, low-earner counting noise and a class allowance, with slightly
wider tails on the side the trophy is moving towards.  Story chapters
are then made monotone (a later chapter cannot be held by more players
than an earlier one) and the platinum is capped at the rarest other
trophy, since its earners are a subset of every other trophy's.

Standard library only, deterministic, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Per class: prior slope of logit(rate) per unit log-age, soft ceiling for
# the median rate a week in (percent), how far newly tracked players get
# relative to the tracked pool's current rate (for the dilution regime),
# and extra logit sigma for hard-to-call classes.
CLASS_PARAMS = {
    "platinum":   {"slope": 0.05, "ceiling": 9.0,   "alpha": 0.01, "sigma": 0.20},
    "completion": {"slope": 0.30, "ceiling": 25.0,  "alpha": 0.02, "sigma": 0.16},
    "grind":      {"slope": 0.45, "ceiling": 45.0,  "alpha": 0.06, "sigma": 0.11},
    "story_main": {"slope": 1.15, "ceiling": 94.0,  "alpha": 0.30, "sigma": 0.04},
    "story_side": {"slope": 0.95, "ceiling": 88.0,  "alpha": 0.22, "sigma": 0.06},
    "challenge":  {"slope": 0.70, "ceiling": 75.0,  "alpha": 0.22, "sigma": 0.07},
    "activity":   {"slope": 0.65, "ceiling": 75.0,  "alpha": 0.28, "sigma": 0.07},
}

CHAPTER_RE = re.compile(r"chapter\s+(\d+)")


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def classify(trophy):
    """Bucket a trophy by how its unlock pattern behaves over a launch week."""
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if re.search(r"complete\s+\d+", desc):
        return "grind"
    if CHAPTER_RE.search(desc) or "complete the final mission." in desc:
        return "story_main"
    if desc.startswith("complete "):
        return "story_side"
    if any(w in desc for w in ("win", "earn", "beat", "take down", "rob", "lose a")):
        return "challenge"
    return "activity"


def chapter_order(trophy):
    """Position inside the main story chain, or None when not in the chain."""
    desc = (trophy.get("description") or "").lower()
    m = CHAPTER_RE.search(desc)
    if m:
        return int(m.group(1))
    if "complete the final mission." in desc:
        return 90
    return None


def ceiling_for(trophy, cls):
    """Soft week-one ceiling for a class, tighter for later story steps."""
    base = CLASS_PARAMS[cls]["ceiling"]
    if cls == "story_main":
        order = chapter_order(trophy)
        if order is not None and order >= 90:
            return 72.0
        if order is not None:
            return max(78.0, 95.0 - 3.0 * order)
    return base


def alpha_for(trophy, cls):
    """Relative earn rate of a newly tracked cohort, for the dilution mix."""
    base = CLASS_PARAMS[cls]["alpha"]
    if cls == "story_main":
        order = chapter_order(trophy)
        if order == 1:
            return 0.50
        if order == 2:
            return 0.36
        if order == 3:
            return 0.26
        if order == 4:
            return 0.18
        if order is not None:
            return 0.10
    return base


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x; returns (a, b, resid_sd, se_b)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0)
    var = sum(w * r * r for r, w in zip(resid, ws)) / dof
    se_b = math.sqrt(var / sxx) if var > 0 else 0.0
    return (a, b, math.sqrt(var), se_b)


def snapshots(case, anchor):
    """Collapse the read history to distinct page states, oldest first.

    Each entry: (age_hours, log_age, players_tracked, rates list).
    A state repeats verbatim until Exophase refreshes, so keeping every
    read would overweight stale stretches and flatten any fitted slope.
    """
    out = []
    for read in case["history"]:
        key_rates = read["rates"]
        n = read["players_tracked"]
        if out and out[-1][2] == n and out[-1][3] == key_rates:
            continue
        t = parse_time(read.get("slot") or read["at"])
        age = max(0.5, (t - anchor) / 3600.0 + 2.0)
        out.append((age, math.log(age), n, key_rates))
    return out


def stale_forecast(rate, earners, players, cls, trophy, horizon_h, age_h):
    """Median logit and sigma when the page has barely refreshed.

    The next refresh is expected to multiply the tracked pool several
    times with players who hold each trophy at a class-relative rate,
    while the already tracked cohort keeps progressing meanwhile.
    """
    if rate >= 99.9:
        return logit(99.95), 0.35
    stale_h = max(1.0, age_h)
    p_refresh = 0.88 if stale_h >= 3.0 else 0.65
    mult = 5.0 + min(7.0, horizon_h / 24.0)
    alpha = alpha_for(trophy, cls)
    growth = 1.0 + 0.35 * min(1.0, horizon_h / 6.0) * (
        1.0 if cls.startswith("story") else 0.45
    )
    r = max(rate, 0.01)
    diluted = (earners * growth + alpha * (r / 100.0) * (mult - 1.0) * players) / (
        mult * players
    ) * 100.0
    grown = min(99.0, r * growth)
    med = p_refresh * logit(diluted) + (1.0 - p_refresh) * logit(grown)
    sigma = 0.95 if stale_h >= 3.0 else 0.70
    return med, sigma


def trend_forecast(series, snaps, cls, trophy, x_last, x_target, horizon_days):
    """Median logit, sigma and used slope for a history with real refreshes.

    series is this trophy's (log_age, logit_rate) over the snapshots.
    """
    last_age = snaps[-1][0]
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    # Recent window: snapshots from the last 30 hours of age, recency weighted.
    rec = [(p[0], p[1], s) for p, s in zip(series, snaps) if last_age - s[0] <= 30.0]
    w_rec = [
        (0.5 ** ((last_age - s[0]) / 18.0)) * (0.3 + 0.7 * s[2] / snaps[-1][2])
        for _, _, s in rec
    ]
    fit_rec = wls([p[0] for p in rec], [p[1] for p in rec], w_rec)
    w_all = [
        (0.5 ** ((last_age - s[0]) / 36.0)) * (0.3 + 0.7 * s[2] / snaps[-1][2])
        for s in snaps
    ]
    fit_all = wls(xs, ys, w_all)
    params = CLASS_PARAMS[cls]
    earners = max(0.0, series[-1][2])
    # Reliability of this trophy's own slope: more earners and more states
    # mean the slope is the trophy's, otherwise it is the class prior's.
    rel = (earners / (earners + 30.0)) * min(1.0, 0.35 + len(series) / 9.0)
    slopes = []
    if fit_rec:
        slopes.append((fit_rec[1], 0.65))
    if fit_all:
        slopes.append((fit_all[1], 0.35))
    if slopes:
        raw_slope = sum(b * w for b, w in slopes) / sum(w for _, w in slopes)
    else:
        raw_slope = params["slope"]
    raw_slope = min(2.4, max(-1.4, raw_slope))
    slope = rel * raw_slope + (1.0 - rel) * params["slope"]
    slope = min(2.2, max(-1.2, slope))
    # Anchor on a smoothed last logit (last three states, latest heaviest).
    tail = ys[-3:]
    tw = [0.2, 0.3, 0.5][-len(tail):]
    last_logit = sum(y * w for y, w in zip(tail, tw)) / sum(tw)
    gap = x_target - x_last
    increment = slope * gap
    if increment > 2.6:
        increment = 2.6 + 0.35 * (increment - 2.6)
    if increment < -2.0:
        increment = -2.0 + 0.35 * (increment + 2.0)
    med = last_logit + increment
    # Blend a little of the regression line's own level at the target: it
    # resists a single odd last state, the anchor resists a bent line.
    if fit_rec:
        line = fit_rec[0] + fit_rec[1] * x_target
        med = 0.78 * med + 0.22 * line
    resid = fit_rec[2] if fit_rec else 0.3
    se_part = min(0.55, (fit_rec[3] if fit_rec else 0.3) * gap)
    base = 0.34 * math.sqrt(max(horizon_days, 0.05))
    count_noise = min(0.5, 0.85 / math.sqrt(max(earners, 1.0)))
    sigma = math.sqrt(
        base * base
        + (0.55 * min(0.65, resid)) ** 2
        + se_part * se_part
        + count_noise * count_noise
        + params["sigma"] ** 2
    )
    sigma = min(1.30, max(0.32, sigma))
    return med, sigma, slope, last_logit


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    anchor = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    snaps = snapshots(case, anchor)
    at_ts = parse_time(case["at"])
    target_ts = parse_time(case["target"])
    age_at = max(0.5, (at_ts - anchor) / 3600.0 + 2.0)
    age_target = max(age_at + 0.05, (target_ts - anchor) / 3600.0 + 2.0)
    x_last = snaps[-1][1]
    x_target = math.log(age_target)
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    n_last = snaps[-1][2]
    span_h = snaps[-1][0] - snaps[0][0] if len(snaps) > 1 else 0.0

    medians = []
    sigmas = []
    slopes = []
    for i, trophy in enumerate(trophies):
        cls = classify(trophy)
        series = []
        for age, lx, n, rates in snaps:
            r = rates[i]
            if r is None:
                continue
            series.append((lx, logit(r), r / 100.0 * n))
        rate_last = None
        for _, _, rates in [(s[0], s[1], s[3]) for s in snaps]:
            if rates[i] is not None:
                rate_last = rates[i]
        if rate_last is None:
            rate_last = 0.01
        earners = rate_last / 100.0 * n_last
        if rate_last >= 99.9:
            medians.append(logit(99.95))
            sigmas.append(0.30)
            slopes.append(0.0)
            continue
        med_b, sig_b = stale_forecast(
            rate_last, earners, n_last, cls, trophy, horizon_h, age_at
        )
        w_trend = 0.0
        if len(snaps) >= 3:
            w_trend = min(1.0, (len(snaps) - 2) / 3.0) * min(1.0, span_h / 24.0)
        if w_trend > 0.0:
            med_t, sig_t, slope_t, _ = trend_forecast(
                series, snaps, cls, trophy, x_last, x_target, horizon_days
            )
            med = w_trend * med_t + (1.0 - w_trend) * med_b
            sig = math.sqrt((w_trend * sig_t) ** 2 + ((1.0 - w_trend) * sig_b) ** 2)
            sig += 0.20 * w_trend * (1.0 - w_trend) * abs(med_t - med_b)
            slope_used = slope_t
        else:
            med, sig, slope_used = med_b, sig_b, 0.0
        # Soft class ceiling: a median may pass it only when the last rate
        # is already up there, in which case the ceiling rides above it.
        ceiling = ceiling_for(trophy, cls)
        ceil_logit = max(logit(ceiling), logit(rate_last) + 0.7)
        if med > ceil_logit:
            med = ceil_logit + 0.35 * (med - ceil_logit)
        # Dilution floor: earners stay earned; even a sixfold larger pool
        # cannot push the rate below earners over that pool.
        if earners >= 2.0:
            floor_pct = earners / (n_last * 6.0) * 100.0
            med = max(med, logit(floor_pct))
        medians.append(med)
        sigmas.append(sig)
        slopes.append(slope_used)

    # Story chain coherence: medians non-increasing along the main chain.
    chain = []
    for i, trophy in enumerate(trophies):
        order = chapter_order(trophy)
        if order is not None:
            chain.append((order, i))
    chain.sort()
    idx = [i for _, i in chain]
    if len(idx) >= 2:
        vals = [medians[i] for i in idx]
        wts = [1.0] * len(idx)
        # Pool adjacent violators, decreasing target, plain average pools.
        blocks = [[v, w, 1] for v, w in zip(vals, wts)]
        changed = True
        while changed:
            changed = False
            for k in range(len(blocks) - 1):
                if blocks[k][0] < blocks[k + 1][0]:
                    tot_w = blocks[k][1] + blocks[k + 1][1]
                    avg = (
                        blocks[k][0] * blocks[k][1] + blocks[k + 1][0] * blocks[k + 1][1]
                    ) / tot_w
                    blocks[k:k + 2] = [[avg, tot_w, blocks[k][2] + blocks[k + 1][2]]]
                    changed = True
                    break
        pooled = []
        for avg, _, count in blocks:
            pooled.extend([avg] * count)
        for pos, i in enumerate(idx):
            medians[i] = pooled[pos]
    # Platinum earners are a subset of every other trophy's earners.
    for i, trophy in enumerate(trophies):
        if trophy.get("tier") == "platinum" and len(trophies) > 1:
            cap = min(medians[j] for j in range(len(trophies)) if j != i)
            if medians[i] > cap:
                medians[i] = cap

    rows = []
    for i, trophy in enumerate(trophies):
        med = medians[i]
        sig = sigmas[i]
        slope = slopes[i]
        hi_scale = 1.0 + 0.10 * max(0.0, slope)
        lo_scale = 1.0 + 0.10 * max(0.0, -slope)
        qs = []
        for z in Z:
            scale = hi_scale if z > 0 else lo_scale
            q = inv_logit(med + z * sig * scale)
            qs.append(round4(min(100.0, max(0.01, q))))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
