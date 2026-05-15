ALL_CATEGORIES = [
    "Cell_Phones_and_Accessories",
    "Beauty_and_Personal_Care",
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Grocery_and_Gourmet_Food",
    "Appliances",
    "Home_and_Kitchen",
    "Baby_Products",
    "Sports_and_Outdoors",
    "Video_Games",
]

CATEGORY_MAP = {
    "Cell_Phones_and_Accessories": {
        "jumia": "phones-tablets",
        "konga": "phones-tablets-5294",
    },
    "Beauty_and_Personal_Care": {
        "jumia": "health-beauty",
        "konga": "health-beauty-2759",
    },
    "Electronics": {
        "jumia": "electronics",
        "konga": "electronics-5261",
    },
    "Clothing_Shoes_and_Jewelry": {
        "jumia": "category-fashion-by-jumia",
        "konga": "fashion-8",
    },
    "Grocery_and_Gourmet_Food": {
        "jumia": "supermarket",
        "konga": "food-drinks-grocery-3198",
    },
    "Appliances": {
        "jumia": "appliances",
        "konga": "large-appliances-3144",
    },
    "Home_and_Kitchen": {
        "jumia": "home-office",
        "konga": "home-kitchen-6",
    },
    "Baby_Products": {
        "jumia": "baby-products",
        "konga": "baby-kids-3155",
    },
    "Sports_and_Outdoors": {
        "jumia": "sporting-goods",
        "konga": "sports-fitness-2766",
    },
    "Video_Games": {
        "jumia": "video-games",
        "konga": "gaming-4518",
    },
}

BATCH_SCHEDULE = [
    [
        ("jumia", "Cell_Phones_and_Accessories"),
        ("jumia", "Beauty_and_Personal_Care"),
        ("konga", "Cell_Phones_and_Accessories"),
        ("konga", "Beauty_and_Personal_Care"),
    ],
    [
        ("jumia", "Electronics"),
        ("jumia", "Clothing_Shoes_and_Jewelry"),
        ("konga", "Electronics"),
        ("konga", "Clothing_Shoes_and_Jewelry"),
    ],
    [
        ("jumia", "Grocery_and_Gourmet_Food"),
        ("jumia", "Appliances"),
        ("jumia", "Home_and_Kitchen"),
        ("konga", "Grocery_and_Gourmet_Food"),
        ("konga", "Appliances"),
    ],
    [
        ("jumia", "Baby_Products"),
        ("konga", "Home_and_Kitchen"),
        ("konga", "Baby_Products"),
    ],
    [
        ("jumia", "Sports_and_Outdoors"),
        ("jumia", "Video_Games"),
        ("konga", "Sports_and_Outdoors"),
        ("konga", "Video_Games"),
    ],
]

AMAZON_2023_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"
AMAZON_2023_URLS = {
    category: f"{AMAZON_2023_BASE}/{category}.jsonl.gz"
    for category in ALL_CATEGORIES
}
