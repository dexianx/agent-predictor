from optimized_scraper.platforms import jumia


def test_parse_jumia_listing_from_store():
    html = """
    <script>window.__STORE__={"products":[{"sku":"IT724MP7ZRM7NNAFAMZ","name":"City 200","displayName":"itel City 200","brand":"itel","categories":["Phones & Tablets","Mobile Phones","Smartphones"],"prices":{"rawPrice":"138710.00","price":"₦ 138,710"},"rating":{"average":4.1,"totalRatings":16},"url":"/itel-city-200-419283338.html"}]};</script>
    """
    products = jumia.parse_listing_page(html)
    assert len(products) == 1
    assert products[0].url == "https://www.jumia.com.ng/itel-city-200-419283338.html"
    assert products[0].brand == "itel"
    assert products[0].price_ngn == 138710.0
    assert products[0].review_count == 16


def test_parse_jumia_product_reviews_from_html_and_jsonld():
    html = """
    <script>window.__STORE__={"products":[{"sku":"IT724MP7ZRM7NNAFAMZ","displayName":"itel City 200","brand":"itel","categories":["Phones & Tablets","Android Phones"],"prices":{"rawPrice":"138710.00"},"rating":{"average":4.1,"totalRatings":16},"url":"/itel-city-200-419283338.html"}]};</script>
    <section><div id="feedback"></div><article><div class="stars">5 out of 5</div><h3>Itel City 200</h3><p class="-pvm">Words can't explain. This phone is a MACHINE</p><div><span>12-05-2026</span><span>by Ugochukwu</span></div><div>Verified Purchase</div></article></section>
    <script type="application/ld+json">{"@type":"Product","review":[{"@type":"Review","reviewRating":{"ratingValue":2},"datePublished":"12-05-2026","name":"Slow","reviewBody":"Too slow to game with","author":{"name":"by Esebonu"}}]}</script>
    """
    product = jumia._product_from_store(html, "https://www.jumia.com.ng/itel-city-200-419283338.html")
    reviews = jumia._dedup_reviews(jumia._reviews_from_feedback_html(html) + jumia._reviews_from_json_ld(html))
    assert product.sku == "IT724MP7ZRM7NNAFAMZ"
    assert product.price_ngn == 138710.0
    assert len(reviews) == 2
    assert reviews[0].verified_purchase is True
