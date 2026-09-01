"""Canonical country centroids and distance calculations used by producer and detector."""

import math

COUNTRY_CENTROIDS = {
    "US": (39.8, -98.6),
    "GB": (54.0, -2.0),
    "DE": (51.2, 10.4),
    "FR": (46.6, 2.2),
    "JP": (36.2, 138.3),
    "AU": (-25.3, 133.8),
    "BR": (-14.2, -51.9),
    "IN": (20.6, 79.0),
    "CA": (56.1, -106.3),
    "SG": (1.35, 103.8),
    "NL": (52.1, 5.3),
    "MX": (23.6, -102.5),
    "ZA": (-30.6, 22.9),
    "AE": (23.4, 53.8),
    "KR": (35.9, 127.8),
}


def haversine_km(country_a: str, country_b: str) -> float:
    """Distance between configured country centroids, or zero for unknown codes."""
    if country_a not in COUNTRY_CENTROIDS or country_b not in COUNTRY_CENTROIDS:
        return 0.0
    lat1, lon1 = COUNTRY_CENTROIDS[country_a]
    lat2, lon2 = COUNTRY_CENTROIDS[country_b]
    radius_km = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(value))
