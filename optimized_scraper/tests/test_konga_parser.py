from optimized_scraper.platforms import konga


def test_parse_konga_listing_from_next_data():
    html = """
    <script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"items":[{"name":"Tecno Spark","sku":"KG2991823","url_key":"/product/tecno-spark-2991823","special_price":"85000","average_rating":4.2,"total_reviews":12,"brand":"Tecno"}]}}}</script>
    """
    products = konga.parse_listing_page(html)
    assert len(products) == 1
    assert products[0].url == "https://www.konga.com/product/tecno-spark-2991823"
    assert products[0].sku == "KG2991823"
    assert products[0].review_count == 12


def test_parse_konga_jsonld_product():
    html = """
    <script type="application/ld+json">{"@type":"Product","sku":"KG2991823","name":"Tecno Spark","brand":{"name":"Tecno"},"offers":{"price":"85000"},"aggregateRating":{"ratingValue":"4.2","reviewCount":"12"},"review":[{"reviewRating":{"ratingValue":"5"},"name":"Great","reviewBody":"Battery dey last well well","author":{"name":"Ada"},"datePublished":"March 15, 2024"}]}</script>
    """
    product, reviews = konga._product_from_json_ld(html, "https://www.konga.com/product/tecno-spark-2991823")
    assert product.sku == "KG2991823"
    assert product.brand == "Tecno"
    assert product.price_ngn == 85000.0
    assert len(reviews) == 1
    assert reviews[0].text == "Battery dey last well well"
