# -*- coding: utf-8 -*-
from zvt.api.selector import get_big_players, get_player_success_rate
from zvt.domain import DragonAndTiger
from zvt.utils.time_utils import date_time_by_interval, current_date

if __name__ == "__main__":
    # 🧠 ML Signal: Recording data with a specific provider could indicate a pattern of data source preference.
    provider = "em"
    DragonAndTiger.record_data(provider=provider)
    # ✅ Best Practice: Use descriptive variable names for clarity.
    end_timestamp = date_time_by_interval(current_date(), -60)
    # recent year
    # ✅ Best Practice: Use descriptive variable names for clarity.
    start_timestamp = date_time_by_interval(end_timestamp, -400)
    # ✅ Best Practice: Use f-strings for more readable and efficient string formatting.
    print(f"{start_timestamp} to {end_timestamp}")
    players = get_big_players(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    print(players)
    # 🧠 ML Signal: Calculating player success rates over intervals could indicate a pattern of performance analysis.
    # 🧠 ML Signal: Fetching big players within a time range could indicate a pattern of interest in specific market participants.
    df = get_player_success_rate(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp, intervals=[3, 5, 10], players=players
    )
    print(df)