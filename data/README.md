# Local data directory

Personal exports and generated personal artifacts are intentionally excluded.

Expected local layout:

```text
data/
├── raw/
│   ├── lastfm/
│   │   └── recenttracks.csv
│   └── musicbrainz/
├── interim/
└── processed/
```

Do not commit personal Last.fm exports, canonical tables, MusicBrainz caches, personal graphs, or inferred metadata by default.
