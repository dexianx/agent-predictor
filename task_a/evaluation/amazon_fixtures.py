"""Amazon-only eval fixtures: synthetic products not in the KG, with related proxy items."""

from dataclasses import dataclass

# Persona from personas.db / Neo4j (amazon_us, locale US)
EVAL_USER_ID = "AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"


@dataclass(frozen=True)
class UnknownProductCase:
    """Product absent from Neo4j; category peers + proxy item supply eval references."""

    label: str
    item_name: str
    item_category: str
    item_brand: str
    item_description: str
    price_usd: float
    proxy_item_id: str
    proxy_item_title: str

    def to_simulate_request(self, user_id: str = EVAL_USER_ID):
        from task_a.inference.task_a import SimulateRequest

        return SimulateRequest(
            user_id=user_id,
            item_name=self.item_name,
            price_ngn=0.0,
            price_usd=self.price_usd,
            item_description=self.item_description,
            item_category=self.item_category,
            item_brand=self.item_brand,
        )


# Each name is synthetic (2025/2026 models) — verified not to title-match KG items.
UNKNOWN_PRODUCT_CASES: tuple[UnknownProductCase, ...] = (
    UnknownProductCase(
        label="headphones · Home Audio",
        item_name="Bose QuietComfort Ultra 2 Wireless Noise-Cancelling Headphones",
        item_category="Home Audio & Theater",
        item_brand="Bose",
        item_description=(
            "Over-ear wireless ANC headphones, 40-hour battery, Bluetooth 5.3, USB-C"
        ),
        price_usd=349.99,
        proxy_item_id="B091K4WYD1",
        proxy_item_title=(
            "OneOdio Over Ear Headphone, Wired Bass Headsets with 50mm Driver, Foldable"
        ),
    ),
    UnknownProductCase(
        label="laptop · Computers",
        item_name="Dell XPS 15 9650 OLED Laptop 2026 Edition Intel Core Ultra 9",
        item_category="Computers",
        item_brand="Dell",
        item_description=(
            "15.6-inch 4K OLED, 64GB RAM, 2TB SSD, NVIDIA RTX 5070, Thunderbolt 5"
        ),
        price_usd=2199.99,
        proxy_item_id="B0C2ZMJW53",
        proxy_item_title=(
            "Acer Aspire 3 A314-23P-R3QA Slim Laptop | 14.0\" Full HD IPS Display"
        ),
    ),
    UnknownProductCase(
        label="streaming stick · Amazon Devices",
        item_name="Amazon Fire TV Stick 4K Max Gen 3 2026 Streaming Device",
        item_category="Amazon Devices",
        item_brand="Amazon",
        item_description=(
            "4K Max streaming, Wi-Fi 6E, Alexa Voice Remote Pro, Dolby Vision Atmos"
        ),
        price_usd=59.99,
        proxy_item_id="B075X8471B",
        proxy_item_title=(
            "Fire TV Stick with Alexa Voice Remote, streaming media player - Previous Generation"
        ),
    ),
    UnknownProductCase(
        label="phone case · Cell Phones",
        item_name="OtterBox Defender Pro MagSafe Case for iPhone 17 Pro Max",
        item_category="Cell Phones & Accessories",
        item_brand="OtterBox",
        item_description=(
            "Multi-layer drop protection, MagSafe compatible, port covers, belt holster ready"
        ),
        price_usd=64.99,
        proxy_item_id="B07CKPPSB4",
        proxy_item_title=(
            "Honsky Thumbs-up Phone Stand for Tablets, E-readers and Smartphones"
        ),
    ),
)
