import re
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
UK_POSTCODE_URL = "https://api.postcodes.io/postcodes/"

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.IGNORECASE)


def _geocode_openmeteo(query: str):
    params = {"name": query, "count": 1, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=15)
    r.raise_for_status()
    results = (r.json().get("results") or [])
    if not results:
        return None
    top = results[0]
    return {
        "label": f"{top.get('name')}, {top.get('country')}",
        "lat": top.get("latitude"),
        "lon": top.get("longitude"),
        "timezone": top.get("timezone") or "auto",
    }


def _geocode_uk_postcode(postcode: str):
    pc = postcode.strip().replace(" ", "")
    r = requests.get(UK_POSTCODE_URL + pc, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    if data.get("status") != 200 or not data.get("result"):
        return None
    res = data["result"]
    return {
        "label": f"{res.get('admin_district','')}, {res.get('country','UK')} ({res.get('postcode','')})".strip(", "),
        "lat": res.get("latitude"),
        "lon": res.get("longitude"),
        "timezone": "Europe/London",
    }


def fetch_current_weather(location: str) -> dict:
    location = (location or "").strip()
    if not location:
        raise ValueError("location is required")

    geo = _geocode_uk_postcode(location) if UK_POSTCODE_RE.match(location) else None
    if not geo:
        geo = _geocode_openmeteo(location)
    if not geo:
        raise ValueError(f"Could not geocode location: {location}")

    params = {
        "latitude": geo["lat"],
        "longitude": geo["lon"],
        "current_weather": True,
        "timezone": geo["timezone"],
    }
    r = requests.get(WEATHER_URL, params=params, timeout=15)
    r.raise_for_status()
    cw = (r.json().get("current_weather") or {})
    return {
        "label": geo["label"],
        "temperature_c": cw.get("temperature"),
        "windspeed_kmh": cw.get("windspeed"),
        "winddirection_deg": cw.get("winddirection"),
        "observed_at": cw.get("time"),
    }