"""FIFA eSoccer Stats Dashboard."""

import re
import requests
import pandas as pd
import numpy as np
import streamlit as st

# --- Config ---
SUPABASE_URL = "https://izsolcawlvpidwnhrxer.supabase.co"
SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6"
    "Iml6c29sY2F3bHZwaWR3bmhyeGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIwNjk3ODk"
    "sImV4cCI6MjA2NzY0NTc4OX0.7w02SBg3KRm4yf3ftVte-yT59r3hcWdHO6uosIUQBao"
)
HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
CUTOFF = "2026-01-15"
BASE_STAKE = 20
TIME_WINDOW_MIN = 90


# ============================================================
# Data fetching
# ============================================================

def fetch_all(table, select, filter_col, cutoff, extra_filters=""):
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        url = (
            f"{SUPABASE_URL}/rest/v1/{table}"
            f"?select={select}"
            f"&{filter_col}=gte.{cutoff}"
            f"&order={filter_col}"
            f"&offset={offset}&limit={page_size}"
            f"{extra_filters}"
        )
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        if len(data) < page_size:
            break
        offset += page_size
    return all_rows


# ============================================================
# Player name extraction & line parsing
# ============================================================

def extract_players(game):
    m = re.match(r".*\(([^)]+)\)\s*v\s*.*\(([^)]+)\)", str(game))
    if not m:
        return pd.Series([None, None])
    return pd.Series([m.group(1).lower().strip(), m.group(2).lower().strip()])


def parse_line(line_str):
    s = str(line_str).strip()
    if "," in s:
        parts = s.split(",")
        return (float(parts[0].strip()) + float(parts[1].strip())) / 2
    return float(s)


# ============================================================
# Settlement — brother's numpy logic (no if statements)
# ============================================================

def settle_over_bet(over_odds, market, total_goals, line_type, stake):
    line_type = np.asarray(line_type)
    over_odds = np.asarray(over_odds, dtype=float)
    market = np.asarray(market, dtype=float)
    total_goals = np.asarray(total_goals, dtype=float)
    stake = np.asarray(stake, dtype=float)

    result = np.zeros_like(total_goals, dtype=float)

    mask = line_type == "h"
    result[mask] = np.where(
        total_goals[mask] > market[mask],
        stake[mask] * (over_odds[mask] - 1),
        -stake[mask],
    )

    mask = line_type == "lq"
    result[mask] = np.select(
        [
            total_goals[mask] > np.floor(market[mask]),
            total_goals[mask] == np.floor(market[mask]),
        ],
        [
            stake[mask] * (over_odds[mask] - 1),
            -0.5 * stake[mask],
        ],
        default=-stake[mask],
    )

    mask = line_type == "uq"
    result[mask] = np.select(
        [
            total_goals[mask] > np.ceil(market[mask]),
            total_goals[mask] == np.ceil(market[mask]),
        ],
        [
            stake[mask] * (over_odds[mask] - 1),
            0.5 * stake[mask] * (over_odds[mask] - 1),
        ],
        default=-stake[mask],
    )

    mask = line_type == "w"
    result[mask] = np.select(
        [
            total_goals[mask] > market[mask],
            total_goals[mask] < market[mask],
        ],
        [
            stake[mask] * (over_odds[mask] - 1),
            -stake[mask],
        ],
        default=0,
    )

    return result


def settle_under_bet(under_odds, market, total_goals, line_type, stake):
    line_type = np.asarray(line_type)
    under_odds = np.asarray(under_odds, dtype=float)
    market = np.asarray(market, dtype=float)
    total_goals = np.asarray(total_goals, dtype=float)
    stake = np.asarray(stake, dtype=float)

    result = np.zeros_like(total_goals, dtype=float)

    mask = line_type == "h"
    result[mask] = np.where(
        total_goals[mask] < market[mask],
        stake[mask] * (under_odds[mask] - 1),
        -stake[mask],
    )

    mask = line_type == "lq"
    result[mask] = np.select(
        [
            total_goals[mask] < np.floor(market[mask]),
            total_goals[mask] == np.floor(market[mask]),
        ],
        [
            stake[mask] * (under_odds[mask] - 1),
            stake[mask] * 0.5 * (under_odds[mask] - 1),
        ],
        default=-stake[mask],
    )

    mask = line_type == "uq"
    result[mask] = np.select(
        [
            total_goals[mask] < np.ceil(market[mask]),
            total_goals[mask] == np.ceil(market[mask]),
        ],
        [
            stake[mask] * (under_odds[mask] - 1),
            -0.5 * stake[mask],
        ],
        default=-stake[mask],
    )

    mask = line_type == "w"
    result[mask] = np.select(
        [
            total_goals[mask] < market[mask],
            total_goals[mask] > market[mask],
        ],
        [
            stake[mask] * (under_odds[mask] - 1),
            -stake[mask],
        ],
        default=0,
    )

    return result


# ============================================================
# Pipeline
# ============================================================

@st.cache_data(ttl=300)
def load_and_compute():
    """Fetch, match, settle. Cached for 5 min."""

    # 1. Fetch
    bets_raw = fetch_all(
        "esoccerbets",
        "id,isOver,actualLine,odds,stake,game,gameStart,league",
        "gameStart", CUTOFF,
        "&game=neq.SCRAPER_JOB",
    )
    battle_raw = fetch_all("fifa_data_predictions", "date,home_player,away_player,total_goals", "date", CUTOFF)
    h2h_raw = fetch_all("fifa_data_h2h", "date,home_player,away_player,total_goals", "date", CUTOFF)
    volta_raw = fetch_all("fifa_data_volta", "date,home_player,away_player,total_goals", "date", CUTOFF)

    bets = pd.DataFrame(bets_raw)
    if bets.empty:
        return None, None, 0, 0

    # 2. Extract player names
    bets[["home_p", "away_p"]] = bets["game"].apply(extract_players)
    bets = bets.dropna(subset=["home_p", "league"])
    bets["gameStart"] = pd.to_datetime(bets["gameStart"], utc=True)
    total_bets_fetched = len(bets)

    # 3. Prepare result tables
    tables = {
        "battle": pd.DataFrame(battle_raw),
        "h2h": pd.DataFrame(h2h_raw),
        "volta": pd.DataFrame(volta_raw),
    }
    for _, df in tables.items():
        if df.empty:
            continue
        df["home_p"] = df["home_player"].str.lower().str.strip()
        df["away_p"] = df["away_player"].str.lower().str.strip()
        df["date"] = pd.to_datetime(df["date"], utc=True)

    # 4. Match bets → results
    parts = []
    for league in ["battle", "h2h", "volta"]:
        lb = bets[bets["league"] == league]
        lr = tables[league]
        if lb.empty or lr.empty:
            continue
        merged = lb.merge(lr, on=["home_p", "away_p"], how="inner")
        merged["time_diff"] = (merged["gameStart"] - merged["date"]).abs()
        merged = merged[merged["time_diff"] <= pd.Timedelta(minutes=TIME_WINDOW_MIN)]
        merged = merged.sort_values("time_diff").drop_duplicates(subset="id", keep="first")
        parts.append(merged)

    if not parts:
        return None, None, total_bets_fetched, 0

    matched = pd.concat(parts, ignore_index=True)

    # 5. Parse lines & classify
    matched["market"] = matched["actualLine"].apply(parse_line)
    m_val = matched["market"]
    matched["line_type"] = np.select(
        [
            np.isclose((m_val + 0.5) % 1, 0, atol=1e-8),
            np.isclose((m_val + 0.25) % 1, 0, atol=1e-8),
            np.isclose((m_val - 0.25) % 1, 0, atol=1e-8),
        ],
        ["h", "uq", "lq"],
        default="w",
    )

    # 6. Settle
    matched["actual_stake"] = BASE_STAKE * matched["stake"]
    over = matched["isOver"] == True  # noqa: E712
    under = ~over

    matched.loc[over, "profit"] = settle_over_bet(
        matched.loc[over, "odds"].values,
        matched.loc[over, "market"].values,
        matched.loc[over, "total_goals"].values,
        matched.loc[over, "line_type"].values,
        matched.loc[over, "actual_stake"].values,
    )
    matched.loc[under, "profit"] = settle_under_bet(
        matched.loc[under, "odds"].values,
        matched.loc[under, "market"].values,
        matched.loc[under, "total_goals"].values,
        matched.loc[under, "line_type"].values,
        matched.loc[under, "actual_stake"].values,
    )

    # 7. Build stats table
    matched["side"] = matched["isOver"].map({True: "over", False: "under"})
    stats = (
        matched.groupby(["league", "side"])
        .agg(num_bets=("id", "count"), total_staked=("actual_stake", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    stats["roi"] = stats["profit"] / stats["total_staked"] * 100

    league_ord = {"battle": 0, "h2h": 1, "volta": 2}
    side_ord = {"over": 0, "under": 1}
    stats = stats.sort_values(
        by=["league", "side"],
        key=lambda col: col.map(league_ord) if col.name == "league" else col.map(side_ord),
    )

    return stats, matched, total_bets_fetched, len(matched)


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="FIFA Stats", layout="centered")
st.title("FIFA eSoccer Stats")

if st.button("Update Stats", type="primary"):
    st.cache_data.clear()

stats, matched, total_fetched, total_matched = load_and_compute()

if stats is not None and not stats.empty:
    st.caption(f"{total_fetched} bets fetched, {total_matched} matched to results")

    # Format stats for display
    display = stats.copy()
    display.columns = ["League", "Side", "# Bets", "Staked", "Profit", "ROI %"]
    display["Staked"] = display["Staked"].apply(lambda x: f"£{x:,.2f}")
    display["Profit"] = display["Profit"].apply(lambda x: f"£{x:,.2f}")
    display["ROI %"] = display["ROI %"].apply(lambda x: f"{x:.1f}%")

    # Total row
    tot_bets = stats["num_bets"].sum()
    tot_staked = stats["total_staked"].sum()
    tot_profit = stats["profit"].sum()
    tot_roi = (tot_profit / tot_staked * 100) if tot_staked > 0 else 0

    total_row = pd.DataFrame([{
        "League": "TOTAL",
        "Side": "",
        "# Bets": tot_bets,
        "Staked": f"£{tot_staked:,.2f}",
        "Profit": f"£{tot_profit:,.2f}",
        "ROI %": f"{tot_roi:.1f}%",
    }])
    display = pd.concat([display, total_row], ignore_index=True)

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Example bets
    with st.expander("Example settled bets (first 3 per league)"):
        for league in ["battle", "h2h", "volta"]:
            subset = matched[matched["league"] == league].head(3)
            if subset.empty:
                continue
            st.markdown(f"**{league}**")
            for _, row in subset.iterrows():
                side = "OVER" if row["isOver"] else "UNDER"
                st.text(
                    f"  {row['game']} | {side} {row['actualLine']} "
                    f"(market={row['market']}, type={row['line_type']}) | "
                    f"odds={row['odds']} | stake=£{row['actual_stake']:.0f} | "
                    f"goals={int(row['total_goals'])} | profit=£{row['profit']:.2f}"
                )
elif total_fetched > 0:
    st.warning(f"{total_fetched} bets fetched but no matches found.")
else:
    st.info("Click **Update Stats** to load data.")
