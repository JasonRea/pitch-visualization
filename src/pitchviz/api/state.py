from cachetools import TTLCache

# TTL caches — MLB roster refreshes hourly, pitch data valid for the day
players_cache:    TTLCache = TTLCache(maxsize=1,   ttl=3600)
pitch_type_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# In-memory run inputs (survives for the lifetime of the process)
render_jobs: dict[int, dict] = {}
