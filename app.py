from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRODUCTS_FILE = os.path.join(BASE_DIR, "products_dataset.csv")
REVIEWS_FILE = os.path.join(BASE_DIR, "reviews_dataset.csv")

analyzer = SentimentIntensityAnalyzer()


# Find a column even if its name has a different spelling
def find_column(df, possible_names):
    columns = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower().strip() in columns:
            return columns[name.lower().strip()]

    return None


# Safely convert a value into a number
def to_number(value, default=0):
    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return default

    return float(number)


# Load both CSV files
def load_datasets():

    if not os.path.exists(PRODUCTS_FILE):
        raise FileNotFoundError("products_dataset.csv not found")

    if not os.path.exists(REVIEWS_FILE):
        raise FileNotFoundError("reviews_dataset.csv not found")

    products = pd.read_csv(PRODUCTS_FILE, dtype={"product_id": str})
    reviews = pd.read_csv(REVIEWS_FILE, dtype={"product_id": str})

    return products, reviews


# Identify and format product information
def get_product_data(df):

    id_col = find_column(df, ["product_id", "id"])
    name_col = find_column(df, ["name", "product_name", "title"])
    category_col = find_column(df, ["category", "product_category"])
    sub_category_col = find_column(df, ["sub_category", "subcategory"])
    price_col = find_column(df, ["price", "product_price"])
    image_col = find_column(
        df, ["image_url", "image", "product_image", "image_link"]
    )
    rating_col = find_column(df, ["rating", "average_rating"])
    reviews_count_col = find_column(df, ["reviews_count", "review_count"])
    desc_col = find_column(df, ["description", "product_description"])
    positive_col = find_column(df, ["positive_sentiment", "positive"])
    neutral_col = find_column(df, ["neutral_sentiment", "neutral"])
    negative_col = find_column(df, ["negative_sentiment", "negative"])
    sales_col = find_column(df, ["sales_history", "sales"])

    result = []

    for index, row in df.iterrows():

        product_id = (
            str(row[id_col]).strip()
            if id_col and pd.notna(row[id_col])
            else str(index + 1)
        )

        image = (
            str(row[image_col]).strip()
            if image_col and pd.notna(row[image_col])
            else ""
        )

        # Keep sales history as a list of numbers
        sales_history = []

        if sales_col and pd.notna(row[sales_col]):
            raw_sales = str(row[sales_col]).split(",")

            sales_history = [
                to_number(value)
                for value in raw_sales
                if str(value).strip() != ""
            ]

        result.append({
            "id": product_id,
            "product_id": product_id,
            "name": str(row[name_col]) if name_col and pd.notna(row[name_col]) else "Unnamed Product",
            "category": str(row[category_col]) if category_col and pd.notna(row[category_col]) else "Other",
            "sub_category": str(row[sub_category_col]) if sub_category_col and pd.notna(row[sub_category_col]) else "",
            "price": to_number(row[price_col]) if price_col else 0,
            "rating": to_number(row[rating_col]) if rating_col else 0,
            "reviews_count": int(to_number(row[reviews_count_col])) if reviews_count_col else 0,
            "image": image,
            "image_url": image,
            "description": str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else "Product information is not available.",
            "positive_sentiment": to_number(row[positive_col]) if positive_col else 0,
            "neutral_sentiment": to_number(row[neutral_col]) if neutral_col else 0,
            "negative_sentiment": to_number(row[negative_col]) if negative_col else 0,
            "sales_history": sales_history
        })

    return result


# Analyse and retrieve reviews belonging to one product
def get_review_data(df, product_id):

    id_col = find_column(df, ["product_id", "productid"])
    name_col = find_column(
        df, ["reviewer_name", "customer_name", "name", "user"]
    )
    city_col = find_column(df, ["city", "location"])
    rating_col = find_column(df, ["rating", "review_rating", "stars"])
    text_col = find_column(
        df, ["review_text", "review", "text", "comment"]
    )

    result = []

    if not id_col or not text_col:
        return result

    for _, row in df.iterrows():

        row_product_id = str(row[id_col]).strip()

        if row_product_id != str(product_id).strip():
            continue

        review_text = (
            str(row[text_col]).strip()
            if pd.notna(row[text_col])
            else ""
        )

        if not review_text or review_text.lower() in ["nan", "none"]:
            continue

        rating = to_number(row[rating_col]) if rating_col else 0

        sentiment_score = analyzer.polarity_scores(review_text)["compound"]

        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        result.append({
            "name": str(row[name_col]) if name_col and pd.notna(row[name_col]) else "Customer",
            "city": str(row[city_col]) if city_col and pd.notna(row[city_col]) else "",
            "rating": rating,
            "text": review_text,
            "review_text": review_text,
            "sentiment": sentiment
        })

    return result


# Homepage
@app.route("/")
def home():
    return render_template("index.html")


# API: Get all products
@app.route("/api/products", methods=["GET"])
def get_products():

    try:
        products_df, reviews_df = load_datasets()
        products = get_product_data(products_df)

        for product in products:

            product_reviews = get_review_data(
                reviews_df, product["id"]
            )

            if product_reviews:
                product["rating"] = round(
                    sum(r["rating"] for r in product_reviews)
                    / len(product_reviews), 1
                )

                product["reviews_count"] = len(product_reviews)

        return jsonify(products)

    except Exception as error:
        return jsonify({"error": str(error)}), 500


# API: Get one product and its reviews
@app.route("/api/products/<product_id>", methods=["GET"])
def get_product_details(product_id):

    try:
        products_df, reviews_df = load_datasets()
        products = get_product_data(products_df)

        product = next(
            (p for p in products if p["id"] == str(product_id).strip()),
            None
        )

        if product is None:
            return jsonify({"error": "Product not found"}), 404

        product_reviews = get_review_data(reviews_df, product_id)

        total = len(product_reviews)

        positive = sum(
            1 for review in product_reviews
            if review["sentiment"] == "Positive"
        )

        neutral = sum(
            1 for review in product_reviews
            if review["sentiment"] == "Neutral"
        )

        negative = sum(
            1 for review in product_reviews
            if review["sentiment"] == "Negative"
        )

        # Use actual review analysis when reviews exist.
        # Otherwise use the sentiment percentages stored in the products CSV.
        if total > 0:
            analysis = {
                "positive": round(positive / total * 100),
                "neutral": round(neutral / total * 100),
                "negative": round(negative / total * 100),
                "total_reviews": total
            }

            product["rating"] = round(
                sum(r["rating"] for r in product_reviews) / total, 1
            )

        else:
            analysis = {
                "positive": product["positive_sentiment"],
                "neutral": product["neutral_sentiment"],
                "negative": product["negative_sentiment"],
                "total_reviews": 0
            }

        product["reviews_count"] = (
            total if total > 0 else product["reviews_count"]
        )

        return jsonify({
            "product": product,
            "reviews": product_reviews,
            "analysis": analysis
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500


# API: Add a new customer review
@app.route("/api/add-review", methods=["POST"])
def add_review():

    try:
        data = request.get_json(silent=True) or {}

        product_id = str(data.get("product_id", "")).strip()
        name = str(data.get("name", "")).strip()
        city = str(data.get("city", "")).strip()
        review_text = str(
            data.get("text", data.get("review_text", ""))
        ).strip()

        rating = data.get("rating")

        if not product_id or not name or not review_text:
            return jsonify({"error": "Name, product and review are required"}), 400

        try:
            rating = int(rating)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid rating"}), 400

        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating must be between 1 and 5"}), 400

        products_df, reviews_df = load_datasets()
        products = get_product_data(products_df)

        if not any(p["id"] == product_id for p in products):
            return jsonify({"error": "Product not found"}), 404

        # Match the existing review CSV columns
        id_col = find_column(reviews_df, ["product_id", "productid"]) or "product_id"
        name_col = find_column(
            reviews_df, ["reviewer_name", "customer_name", "name", "user"]
        ) or "reviewer_name"
        rating_col = find_column(
            reviews_df, ["rating", "review_rating", "stars"]
        ) or "rating"
        text_col = find_column(
            reviews_df, ["review_text", "review", "text", "comment"]
        ) or "review_text"
        city_col = find_column(reviews_df, ["city", "location"])

        new_review = {
            id_col: product_id,
            name_col: name,
            rating_col: rating,
            text_col: review_text
        }

        if city_col:
            new_review[city_col] = city
        elif city:
            new_review["city"] = city

        reviews_df = pd.concat(
            [reviews_df, pd.DataFrame([new_review])],
            ignore_index=True
        )

        reviews_df.to_csv(REVIEWS_FILE, index=False)

        return jsonify({
            "message": "Review submitted successfully"
        }), 201

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True)
