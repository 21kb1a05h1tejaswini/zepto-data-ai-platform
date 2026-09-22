import sqlite3
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50  # Fixed project constant

CATEGORIES = [
    ("Travel", "catalogue/category/books/travel_2/index.html"),
    ("Mystery", "catalogue/category/books/mystery_3/index.html"),
    ("Historical Fiction", "catalogue/category/books/historical-fiction_4/index.html"),
    ("Sequential Art", "catalogue/category/books/sequential-art_5/index.html")
]

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_data():
    records = []
    for cat_name, cat_url in CATEGORIES:
        url = BASE_URL + cat_url
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.content, "html.parser")
        articles = soup.find_all("article", class_="product_pod")
        for article in articles:
            title = article.h3.a["title"]
            price_raw = article.find("p", class_="price_color").text
            rating_classes = article.find("p", class_="star-rating")["class"]
            rating_text = [c for c in rating_classes if c != "star-rating"][0]
            avail_raw = article.find("p", class_="instock availability").text.strip()
            records.append({
                "title": title,
                "category": cat_name,
                "price_raw": price_raw,
                "rating_raw": rating_text,
                "avail_raw": avail_raw
            })
    return pd.DataFrame(records)

def clean_data(df):
    def parse_price(val):
        match = re.search(r"[\d.]+", str(val))
        return float(match.group()) if match else None

    df["price_gbp"] = df["price_raw"].apply(parse_price)
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    df["rating"] = df["rating_raw"].map(RATING_MAP).fillna(3).astype(int)
    df["in_stock"] = df["avail_raw"].str.contains("In stock", case=False).astype(int)
    return df

def setup_db(df):
    conn = sqlite3.connect("catalog.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER REFERENCES categories(category_id)
        )
    """)

    unique_cats = sorted(df["category"].unique())
    cat_id_map = {}
    for cat in unique_cats:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
        cat_id_map[cat] = cursor.lastrowid

    df["category_id"] = df["category"].map(cat_id_map)
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row["title"], row["price_gbp"], row["price_inr"], row["rating"], row["in_stock"], row["category_id"]))
    
    conn.commit()
    return conn, cat_id_map

def run_queries(conn):
    q1 = "SELECT DISTINCT rating FROM books ORDER BY rating ASC;"
    print("Query 1 (DISTINCT, ORDER BY):\n", pd.read_sql(q1, conn))

    q2 = "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20.0 AND 40.0 ORDER BY price_gbp ASC LIMIT 5;"
    print("\nQuery 2 (WHERE BETWEEN, ORDER BY, LIMIT):\n", pd.read_sql(q2, conn))

    q3 = "SELECT title, rating FROM books WHERE rating IN (4, 5) LIMIT 5;"
    print("\nQuery 3 (WHERE IN, LIMIT):\n", pd.read_sql(q3, conn))

    q4 = "SELECT category_id, COUNT(*) as book_count FROM books GROUP BY category_id;"
    print("\nQuery 4 (GROUP BY/AGGREGATION):\n", pd.read_sql(q4, conn))

    q5 = """
        SELECT b.book_id, b.title, b.rating, b.price_inr, c.category_name
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_inr DESC
        LIMIT 10;
    """
    df_sql_join = pd.read_sql(q5, conn)
    print("\nQuery 5 (JOIN, ORDER BY, LIMIT):\n", df_sql_join)

    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_cats = pd.read_sql("SELECT * FROM categories", conn)
    df_pd_merge = pd.merge(df_books, df_cats, on="category_id")[["book_id", "title", "rating", "price_inr", "category_name"]]
    df_pd_merge = df_pd_merge.sort_values(by=["rating", "price_inr"], ascending=[False, False]).head(10).reset_index(drop=True)

    print("\nParity Check (SQL Join vs Pandas Merge equal?):", df_sql_join.equals(df_pd_merge))

if __name__ == "__main__":
    raw_df = scrape_data()
    clean_df = clean_data(raw_df)
    conn, _ = setup_db(clean_df)
    run_queries(conn)
    conn.close()
