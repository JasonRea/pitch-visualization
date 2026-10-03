from cachetools import TTLCache

# TTL caches — MLB roster refreshes hourly, pitch data valid for the day
players_cache:    TTLCache = TTLCache(maxsize=1,   ttl=3600)
pitch_type_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# Raw per-outing Statcast dataframe, shared across /pitch-types, /movement,
# /heatmap, and /at-bats so a single outing is only fetched once.
outing_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)
