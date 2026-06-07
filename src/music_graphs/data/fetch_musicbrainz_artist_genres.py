from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_ROOT = "https://musicbrainz.org/ws/2"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RateLimiter:
    """Ensure a minimum delay between HTTP request start times."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 1.0:
            raise ValueError(
                "MusicBrainz requests must be separated by at least "
                "one second."
            )

        self.interval_seconds = interval_seconds
        self.last_request_started_at: float | None = None

    def wait(self) -> None:
        now = time.monotonic()

        if self.last_request_started_at is not None:
            elapsed = now - self.last_request_started_at
            remaining = self.interval_seconds - elapsed

            if remaining > 0:
                time.sleep(remaining)

        self.last_request_started_at = time.monotonic()


def load_unique_mbids(input_csv: Path) -> list[str]:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_csv}"
        )

    artists = pd.read_csv(
        input_csv,
        dtype={"artist_mbid": "string"},
        low_memory=False,
    )

    if "artist_mbid" not in artists.columns:
        raise ValueError(
            "The input table does not contain artist_mbid."
        )

    mbids = (
        artists["artist_mbid"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return sorted(
        {
            mbid
            for mbid in mbids
            if mbid
        }
    )


def cache_file_for_mbid(
    cache_dir: Path,
    mbid: str,
) -> Path:
    return cache_dir / f"{mbid}.json"


def load_cached_json(
    cache_file: Path,
) -> dict[str, Any] | None:
    if not cache_file.exists():
        return None

    try:
        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            return None

        return payload

    except (OSError, json.JSONDecodeError):
        return None


def save_json_atomically(
    payload: dict[str, Any],
    output_file: Path,
) -> None:
    temporary_file = output_file.with_suffix(
        ".json.tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(output_file)


def fetch_artist_json(
    session: requests.Session,
    limiter: RateLimiter,
    mbid: str,
    cache_file: Path,
    max_retries: int,
) -> tuple[str, dict[str, Any] | None, str | None]:
    cached_payload = load_cached_json(cache_file)

    if cached_payload is not None:
        return "cached", cached_payload, None

    url = f"{API_ROOT}/artist/{mbid}"

    parameters = {
        "inc": "genres",
        "fmt": "json",
    }

    last_error: str | None = None

    for attempt in range(max_retries + 1):
        limiter.wait()

        try:
            response = session.get(
                url,
                params=parameters,
                timeout=30,
            )

        except requests.RequestException as error:
            last_error = (
                f"{type(error).__name__}: {error}"
            )

            if attempt < max_retries:
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue

            return "failed", None, last_error

        if response.status_code == 200:
            try:
                payload = response.json()

            except requests.JSONDecodeError as error:
                last_error = (
                    f"Invalid JSON response: {error}"
                )

                if attempt < max_retries:
                    time.sleep(min(60, 2 ** (attempt + 1)))
                    continue

                return "failed", None, last_error

            save_json_atomically(
                payload,
                cache_file,
            )

            return "fetched", payload, None

        if response.status_code == 404:
            return (
                "not_found",
                None,
                "MusicBrainz returned HTTP 404.",
            )

        last_error = (
            f"MusicBrainz returned HTTP "
            f"{response.status_code}."
        )

        if (
            response.status_code
            in RETRYABLE_STATUS_CODES
            and attempt < max_retries
        ):
            time.sleep(min(60, 2 ** (attempt + 1)))
            continue

        return "failed", None, last_error

    return "failed", None, last_error


def genre_names_from_payload(
    payload: dict[str, Any] | None,
) -> list[str]:
    if not payload:
        return []

    genres = payload.get("genres", [])

    if not isinstance(genres, list):
        return []

    names = []

    for genre in genres:
        if not isinstance(genre, dict):
            continue

        name = genre.get("name")

        if name:
            names.append(str(name).strip())

    return sorted(
        {
            name
            for name in names
            if name
        }
    )


def write_manifest(
    records: list[dict[str, Any]],
    output_csv: Path,
) -> None:
    manifest = pd.DataFrame.from_records(records)

    if not manifest.empty:
        manifest = manifest.sort_values(
            "artist_mbid"
        ).reset_index(drop=True)

    manifest.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )


def fetch_artist_genres(
    input_csv: Path,
    cache_dir: Path,
    output_dir: Path,
    contact: str,
    delay_seconds: float,
    max_retries: int,
    save_progress_every: int,
) -> None:
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mbids = load_unique_mbids(input_csv)

    print(f"Unique MBIDs to process: {len(mbids):,}")
    print(f"Cache directory: {cache_dir}")
    print(
        "Requests not already cached will be sent "
        f"at intervals of {delay_seconds:.2f} seconds."
    )

    user_agent = (
        f"MusicGraphsResearch/0.1 "
        f"({contact})"
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
    )

    limiter = RateLimiter(
        interval_seconds=delay_seconds
    )

    manifest_output = (
        output_dir
        / "artist_genre_fetch_manifest.csv"
    )

    records: list[dict[str, Any]] = []

    started_at = time.perf_counter()

    for index, mbid in enumerate(
        mbids,
        start=1,
    ):
        cache_file = cache_file_for_mbid(
            cache_dir,
            mbid,
        )

        status, payload, error = fetch_artist_json(
            session=session,
            limiter=limiter,
            mbid=mbid,
            cache_file=cache_file,
            max_retries=max_retries,
        )

        genre_names = genre_names_from_payload(
            payload
        )

        records.append(
            {
                "artist_mbid": mbid,
                "status": status,
                "cache_file": str(cache_file),
                "genre_count": len(genre_names),
                "genre_names": "|".join(genre_names),
                "error": error,
            }
        )

        print(
            f"[{index:,}/{len(mbids):,}] "
            f"{mbid} -> {status}; "
            f"genres={len(genre_names)}"
        )

        if (
            save_progress_every > 0
            and index % save_progress_every == 0
        ):
            write_manifest(
                records,
                manifest_output,
            )

    write_manifest(
        records,
        manifest_output,
    )

    manifest = pd.DataFrame.from_records(
        records
    )

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    summary = {
        "input_file": str(input_csv),
        "unique_mbid_count": int(len(mbids)),
        "cache_directory": str(cache_dir),
        "request_delay_seconds": float(
            delay_seconds
        ),
        "status_counts": {
            status: int(count)
            for status, count in (
                manifest["status"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "artists_with_at_least_one_genre": int(
            (manifest["genre_count"] > 0).sum()
        ),
        "artists_without_genres": int(
            (manifest["genre_count"] == 0).sum()
        ),
        "elapsed_seconds": round(
            float(elapsed_seconds),
            6,
        ),
        "outputs": {
            "manifest": str(
                manifest_output
            ),
        },
    }

    with (
        output_dir
        / "artist_genre_fetch_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and cache MusicBrainz artist genre metadata "
            "for artists with resolved MBIDs."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to canonical_artists.csv.",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "data/raw/musicbrainz/artist_genres"
        ),
        help=(
            "Directory for cached JSON API responses."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/interim/musicbrainz"
        ),
        help=(
            "Directory for the fetch manifest and summary."
        ),
    )

    parser.add_argument(
        "--contact",
        type=str,
        required=True,
        help=(
            "Contact information included in the User-Agent, "
            "normally an email address."
        ),
    )

    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.1,
        help=(
            "Minimum interval between HTTP requests. "
            "Default: 1.1 seconds."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retries after transient errors. Default: 4.",
    )

    parser.add_argument(
        "--save-progress-every",
        type=int,
        default=25,
        help=(
            "Write an intermediate manifest every N artists. "
            "Default: 25."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fetch_artist_genres(
        input_csv=args.input_csv,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        contact=args.contact,
        delay_seconds=args.delay_seconds,
        max_retries=args.max_retries,
        save_progress_every=args.save_progress_every,
    )