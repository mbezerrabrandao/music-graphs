# Local data directory

Personal exports and generated personal artifacts are intentionally excluded.

Expected local layout:

```text
data/
+-- raw/
|   +-- lastfm/
|   |   +-- recenttracks-user_a-exportid.csv
|   |   +-- recenttracks-user_b-exportid.csv
|   +-- musicbrainz/
+-- interim/
+-- processed/
```

All CSV files in `data/raw/lastfm/` are loaded together. Session boundaries
are computed independently per inferred user.

Do not commit personal Last.fm exports, canonical tables, MusicBrainz caches, personal graphs, or inferred metadata by default.
