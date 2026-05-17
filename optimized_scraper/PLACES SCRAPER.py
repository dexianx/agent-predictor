import requests
import pandas as pd
import time
import logging
import sys
from typing import Optional

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

GOOGLE_API_KEY = ""  # <-- Paste your NEW key here

CITIES = [
    "Lagos, Nigeria",
    "Abuja, Nigeria",
    "Port Harcourt, Nigeria",
    "Kano, Nigeria",
    "Ibadan, Nigeria",
    "Enugu, Nigeria",
    "Benin City, Nigeria",
]

PLACE_TYPES = [
    "restaurant",
    "shopping_mall",
    "supermarket",
]

MAX_PLACES_PER_QUERY = 60
REQUEST_DELAY        = 0.5
PLACES_CSV           = "nigeria_places.csv"
REVIEWS_CSV          = "nigeria_reviews.csv"

# ── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL = "https://places.googleapis.com/v1"

# ── API KEY VALIDATOR ─────────────────────────────────────────────────────────

STATUS_HELP = {
    "PERMISSION_DENIED": (
        "PERMISSION_DENIED - API key rejected or Places API (New) not enabled.\n"
        "   1. Enable Places API (New):\n"
        "      https://console.cloud.google.com/apis/library/places.googleapis.com\n"
        "   2. Enable Billing on your project:\n"
        "      https://console.cloud.google.com/billing\n"
        "   3. Check key restrictions:\n"
        "      https://console.cloud.google.com/apis/credentials"
    ),
    "RESOURCE_EXHAUSTED": (
        "RESOURCE_EXHAUSTED - Quota exceeded.\n"
        "   Wait 24 hours or upgrade billing."
    ),
}


def validate_api_key() -> bool:
    """Make a minimal test call to confirm the key works."""
    print("\nValidating API key (Places API NEW)...")
    try:
        url = f"{BASE_URL}/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-FieldMask": "places.id,places.displayName"
        }
        body = {
            "textQuery": "restaurant in Lagos Nigeria",
            "regionCode": "NG"
        }
        resp = requests.post(
            url,
            headers=headers,
            json=body,
            params={"key": GOOGLE_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if "places" in data:
            count = len(data.get("places", []))
            print(f"API key is valid! Test search returned {count} results.\n")
            return True

        status = data.get("error", {}).get("status", "UNKNOWN")
        msg = STATUS_HELP.get(status, f"Error: {data}")
        print(msg)
        return False

    except requests.exceptions.ConnectionError:
        print("No internet connection.")
        return False
    except requests.exceptions.Timeout:
        print("Request timed out.")
        return False
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return False


# ── GOOGLE PLACES HELPERS (NEW API) ───────────────────────────────────────────

def text_search(query: str, page_token: Optional[str] = None) -> dict:
    url = f"{BASE_URL}/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.types,places.businessStatus,places.location"
        ),
    }
    body = {
        "textQuery": query,
        "regionCode": "NG",
        "pageSize": 20,  # New API max is 20 per page
    }
    if page_token:
        body["pageToken"] = page_token

    resp = requests.post(
        url,
        headers=headers,
        json=body,
        params={"key": GOOGLE_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def place_details(place_id: str) -> dict:
    # New API uses places/PLACE_ID format
    if not place_id.startswith("places/"):
        place_id = f"places/{place_id}"

    url = f"{BASE_URL}/{place_id}"
    headers = {
        "X-Goog-FieldMask": (
            "id,displayName,formattedAddress,rating,userRatingCount,"
            "nationalPhoneNumber,websiteUri,regularOpeningHours,priceLevel,"
            "reviews,location,types,businessStatus"
        ),
    }
    resp = requests.get(
        url,
        headers=headers,
        params={"key": GOOGLE_API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── MAIN SCRAPER ──────────────────────────────────────────────────────────────

def fetch_places_for_query(query: str) -> list:
    results = []
    page_token = None

    while len(results) < MAX_PLACES_PER_QUERY:
        data = text_search(query, page_token)
        
        # Check for errors
        if "error" in data:
            status = data["error"].get("status", "UNKNOWN")
            log.warning("API error: %s", status)
            break

        places = data.get("places", [])
        if not places:
            log.info("  No results for: %s", query)
            break

        results.extend(places)
        page_token = data.get("nextPageToken")

        if not page_token or len(results) >= MAX_PLACES_PER_QUERY:
            break

        time.sleep(REQUEST_DELAY)

    return results[:MAX_PLACES_PER_QUERY]


def scrape_all():
    places_rows = []
    reviews_rows = []
    seen_ids = set()

    for city in CITIES:
        for ptype in PLACE_TYPES:
            query = f"{ptype} in {city}"
            log.info("Searching: %s", query)

            summaries = fetch_places_for_query(query)
            log.info("  Found %d candidate places", len(summaries))

            for summary in summaries:
                pid = summary.get("id", "").replace("places/", "")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)

                try:
                    detail = place_details(pid)
                except Exception as exc:
                    log.error("  Details failed for %s: %s", pid, exc)
                    continue

                loc = detail.get("location", {})
                hours = detail.get("regularOpeningHours", {})

                places_rows.append({
                    "place_id":        pid,
                    "name":            detail.get("displayName", {}).get("text"),
                    "category":        ptype,
                    "city":            city.replace(", Nigeria", ""),
                    "address":         detail.get("formattedAddress"),
                    "phone":           detail.get("nationalPhoneNumber"),
                    "website":         detail.get("websiteUri"),
                    "rating":          detail.get("rating"),
                    "total_ratings":   detail.get("userRatingCount"),
                    "price_level":     detail.get("priceLevel"),
                    "business_status": detail.get("businessStatus"),
                    "latitude":        loc.get("latitude"),
                    "longitude":       loc.get("longitude"),
                    "opening_hours":   "; ".join(hours.get("weekdayDescriptions", [])) or None,
                    "types":           ", ".join(detail.get("types", [])),
                })

                for rev in detail.get("reviews", []):
                    author = rev.get("authorAttribution", {})
                    text_obj = rev.get("text", {})
                    reviews_rows.append({
                        "place_id":         pid,
                        "place_name":       detail.get("displayName", {}).get("text"),
                        "category":         ptype,
                        "city":             city.replace(", Nigeria", ""),
                        "reviewer_name":    author.get("displayName"),
                        "reviewer_rating":  rev.get("rating"),
                        "review_text":      text_obj.get("text") if isinstance(text_obj, dict) else text_obj,
                        "review_time":      rev.get("relativePublishTimeDescription"),
                        "review_timestamp": rev.get("publishTime"),
                        "reviewer_url":     author.get("uri"),
                        "language":         rev.get("originalLanguageCode"),
                    })

                time.sleep(REQUEST_DELAY)

    return pd.DataFrame(places_rows), pd.DataFrame(reviews_rows)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_NEW_API_KEY_HERE":
        print(
            "\nNo API key set!\n"
            "Replace GOOGLE_API_KEY in the script with your actual key.\n"
            "Get one at:\n"
            "https://developers.google.com/maps/documentation/places/web-service/get-api-key\n"
        )
        sys.exit(1)

    if not validate_api_key():
        print("\nFix the API key issue above, then re-run the script.")
        sys.exit(1)

    log.info("Starting Nigeria places & reviews scraper (NEW API)...")
    places_df, reviews_df = scrape_all()

    places_df.to_csv(PLACES_CSV, index=False, encoding="utf-8-sig")
    reviews_df.to_csv(REVIEWS_CSV, index=False, encoding="utf-8-sig")

    log.info("Done!")
    log.info("  Places  -> %s  (%d rows)", PLACES_CSV, len(places_df))
    log.info("  Reviews -> %s  (%d rows)", REVIEWS_CSV, len(reviews_df))

    if not places_df.empty:
        print("\n-- Places by category --")
        print(places_df.groupby("category")["name"].count().to_string())
        print("\n-- Places by city --")
        print(places_df.groupby("city")["name"].count().to_string())
        print("\n-- Average rating by category --")
        print(places_df.groupby("category")["rating"].mean().round(2).to_string())

    if not reviews_df.empty:
        print(f"\n-- Total reviews scraped: {len(reviews_df)} reviews across {reviews_df['place_id'].nunique()} places --")


if __name__ == "__main__":
    main()
