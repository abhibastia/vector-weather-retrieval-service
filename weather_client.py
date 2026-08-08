"""
Client for the National Weather Service API (api.weather.gov).

Chosen over OpenWeatherMap/CPC because it is free, needs no API key, and
returns genuinely unstructured narrative prose (alert `description` +
`instruction`, forecast `detailedForecast`) rather than numeric fields - which
is what makes it worth embedding at all.

Two quirks of this API drive the design here:

1. It returns 403 without a descriptive User-Agent containing contact info.
   There is no API key, so the User-Agent *is* the identification scheme.
2. Everything is keyed by lat/lon or by an internal (office, gridX, gridY)
   grid point, never by city name. Resolving "Chicago, IL" therefore needs a
   geocoding step, which CITY_COORDINATES handles without a second API
   dependency (see resolve_location).

Coverage is US-only; a non-US location will 404 at /points.
"""

import hashlib
import logging
import os
import re
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov").rstrip("/")
# api.weather.gov asks for a User-Agent identifying the caller, with a contact
# address, and 403s on a bare/absent one. Deliberately NOT defaulted to a real
# address: hard-coding a personal email into a repo publishes it to anyone who
# reads the source. Set NWS_USER_AGENT (see app.yaml / .env.example) to your own
# contact before deploying.
_DEFAULT_USER_AGENT = "vector-weather-retrieval-service (contact-not-configured)"
_USER_AGENT = os.environ.get("NWS_USER_AGENT") or _DEFAULT_USER_AGENT

if _USER_AGENT == _DEFAULT_USER_AGENT:
    logger.warning(
        "NWS_USER_AGENT is not set. api.weather.gov expects a contact address "
        "and may throttle or reject anonymous callers."
    )

_DEFAULT_TIMEOUT = 30

SOURCE_ALERT = "alert"
SOURCE_FORECAST = "forecast"

# Static geocoding table. A dictionary rather than a geocoder API call because
# it adds no dependency, no rate limit, and no extra failure mode on the
# /weather/sync hot path - and this assignment only ever needs a demo-sized set
# of cities. Any location not listed here can still be passed as a raw
# "lat,lon" string (see resolve_location), so the app is not actually limited
# to these 25.
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "chicago, il": (41.8781, -87.6298),
    "austin, tx": (30.2672, -97.7431),
    "new york, ny": (40.7128, -74.0060),
    "los angeles, ca": (34.0522, -118.2437),
    "houston, tx": (29.7604, -95.3698),
    "phoenix, az": (33.4484, -112.0740),
    "philadelphia, pa": (39.9526, -75.1652),
    "san antonio, tx": (29.4241, -98.4936),
    "san diego, ca": (32.7157, -117.1611),
    "dallas, tx": (32.7767, -96.7970),
    "san jose, ca": (37.3382, -121.8863),
    "jacksonville, fl": (30.3322, -81.6557),
    "columbus, oh": (39.9612, -82.9988),
    "charlotte, nc": (35.2271, -80.8431),
    "indianapolis, in": (39.7684, -86.1581),
    "seattle, wa": (47.6062, -122.3321),
    "denver, co": (39.7392, -104.9903),
    "boston, ma": (42.3601, -71.0589),
    "nashville, tn": (36.1627, -86.7816),
    "oklahoma city, ok": (35.4676, -97.5164),
    "miami, fl": (25.7617, -80.1918),
    "new orleans, la": (29.9511, -90.0715),
    "minneapolis, mn": (44.9778, -93.2650),
    "kansas city, mo": (39.0997, -94.5786),
    "tampa, fl": (27.9506, -82.4572),
}

# Matches a raw coordinate pair like "41.8781,-87.6298" (whitespace tolerated).
_LATLON_RE = re.compile(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$")


class UnknownLocationError(ValueError):
    """Raised when a location string is neither a known city nor a lat/lon pair."""


def resolve_location(location: str) -> tuple[float, float]:
    """Turn a location string into (lat, lon).

    Accepts a city from CITY_COORDINATES ("Chicago, IL", case/space
    insensitive) or a raw "lat,lon" pair so callers aren't boxed in by the
    static table.
    """
    if not isinstance(location, str) or not location.strip():
        raise UnknownLocationError(f"Empty location: {location!r}")

    coords = CITY_COORDINATES.get(location.strip().lower())
    if coords:
        return coords

    match = _LATLON_RE.match(location)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise UnknownLocationError(f"Coordinates out of range: {location!r}")
        return lat, lon

    raise UnknownLocationError(
        f"Unknown location {location!r}. Use one of the known cities "
        f"({', '.join(sorted(CITY_COORDINATES))}) or a 'lat,lon' pair."
    )


def _stable_id(*parts: str) -> str:
    """Deterministic short hash, used as a dedup key where the API gives none.

    Alerts carry their own stable `id`, but forecast periods do not - the same
    period re-fetched an hour later is the same logical document. Hashing
    location + period identity gives ON CONFLICT something to match on so
    re-running /weather/sync updates in place instead of duplicating.
    """
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


class WeatherClient:
    """Thin wrapper around api.weather.gov with the required User-Agent."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                # NWS serves GeoJSON by default but is explicit about wanting
                # the versioned media type; this pins the response shape.
                "Accept": "application/geo+json",
            }
        )
        # /points results are immutable for a given coordinate, so cache them
        # per-client to avoid re-resolving the same city on every sync.
        self._point_cache: dict[tuple[float, float], dict] = {}

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_point(self, lat: float, lon: float) -> dict:
        """Resolve lat/lon to an NWS grid point via GET /points/{lat},{lon}.

        Coordinates are rounded to 4dp because the API rejects higher precision
        with a 301 redirect to the truncated form.
        """
        key = (round(lat, 4), round(lon, 4))
        if key in self._point_cache:
            return self._point_cache[key]

        data = self.get(f"/points/{key[0]},{key[1]}")
        props = data.get("properties", {})
        self._point_cache[key] = props
        return props

    def get_active_alerts(self, lat: float, lon: float) -> list[dict]:
        """Active alerts covering a specific point.

        Uses ?point= rather than ?area={state}: a state-wide query returns
        alerts for counties hundreds of miles from the requested city, which
        would attach wrong-location narrative text to the document.
        """
        data = self.get("/alerts/active", params={"point": f"{round(lat, 4)},{round(lon, 4)}"})
        return data.get("features", []) or []

    def get_forecast(self, lat: float, lon: float) -> dict:
        """Multi-period narrative forecast for a point.

        Follows the `forecast` URL from /points rather than rebuilding
        /gridpoints/{office}/{x},{y}/forecast by hand, so office/grid changes
        on the NWS side can't break this.
        """
        props = self.get_point(lat, lon)
        forecast_url = props.get("forecast")
        if not forecast_url:
            raise ValueError(f"No forecast URL for point {lat},{lon}")

        resp = self._session.get(forecast_url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("properties", {}) or {}

    # -- Normalization ----------------------------------------------------

    def iter_documents(self, location: str, limit: int = 50) -> Iterator[dict]:
        """Yield normalized weather documents for one location.

        Emits alerts first (they are the higher-signal, more urgent text),
        then forecast periods, capped at `limit` documents total.
        """
        lat, lon = resolve_location(location)
        emitted = 0

        for feature in self.get_active_alerts(lat, lon):
            if emitted >= limit:
                return
            doc = self._normalize_alert(location, feature)
            if doc:
                emitted += 1
                yield doc

        if emitted >= limit:
            return

        forecast = self.get_forecast(lat, lon)
        # Both keys are absent on some responses; `updated` in particular does
        # NOT exist on this endpoint despite appearing in older NWS docs.
        issued_at = forecast.get("updateTime") or forecast.get("generatedAt")

        for period in forecast.get("periods", []) or []:
            if emitted >= limit:
                return
            doc = self._normalize_forecast_period(location, period, issued_at)
            if doc:
                emitted += 1
                yield doc

    @staticmethod
    def _normalize_alert(location: str, feature: dict) -> dict | None:
        """Flatten one alert GeoJSON feature into a document record."""
        props = feature.get("properties", {}) or {}

        # `description` explains the hazard, `instruction` says what to do
        # about it. Both are free prose and both are worth retrieving, so they
        # are concatenated into a single narrative rather than embedded apart.
        description = (props.get("description") or "").strip()
        instruction = (props.get("instruction") or "").strip()
        narrative = "\n\n".join(part for part in (description, instruction) if part)
        if not narrative:
            return None

        alert_id = props.get("id") or feature.get("id")
        if not alert_id:
            return None

        return {
            "id": str(alert_id),
            "location": location,
            "source_type": SOURCE_ALERT,
            "headline": props.get("headline") or props.get("event"),
            "event": props.get("event"),
            "narrative_text": narrative,
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective") or props.get("onset"),
            "payload": feature,
        }

    @staticmethod
    def _normalize_forecast_period(
        location: str, period: dict, issued_at: str | None
    ) -> dict | None:
        """Flatten one forecast period into a document record."""
        detailed = (period.get("detailedForecast") or "").strip()
        if not detailed:
            return None

        name = period.get("name") or f"Period {period.get('number')}"
        short = period.get("shortForecast") or ""
        start_time = period.get("startTime") or ""

        # Prefixing the period name and short summary gives the embedding some
        # temporal and categorical anchoring - "Tonight" / "Saturday" and
        # "Slight Chance Showers And Thunderstorms" are exactly the terms a
        # natural-language query like "rain this weekend" will reach for, and
        # detailedForecast alone doesn't always contain them.
        narrative = f"{location} - {name}: {short}. {detailed}".strip()

        return {
            # startTime rather than the period number: period 1 is "Tonight"
            # today and something else tomorrow, so numbering alone would make
            # re-syncs overwrite unrelated periods.
            "id": _stable_id("forecast", location, start_time, name),
            "location": location,
            "source_type": SOURCE_FORECAST,
            "headline": f"{name}: {short}" if short else name,
            "event": short or None,
            "narrative_text": narrative,
            "issued_at": issued_at,
            "effective_at": start_time or None,
            "payload": period,
        }
