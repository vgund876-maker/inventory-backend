import io
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import open_clip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight CPU-optimized OpenCLIP ViT-B-32
device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def extract_vector(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()

@app.get("/")
def health_check():
    return {"status": "AI Backend Online"}

@app.post("/api/products/add")
async def add_product(
    name: str = Form(...),
    category: str = Form("General"),
    price_good: float = Form(...),
    price_medium: float = Form(...),
    price_low: float = Form(...),
    stock: int = Form(10),
    images: list[UploadFile] = File(...)
):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO products (name, category, price_good, price_medium, price_low, stock)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
        """, (name, category, price_good, price_medium, price_low, stock))
        product_id = cur.fetchone()[0]

        for img in images:
            raw = await img.read()
            vector = extract_vector(raw)
            cur.execute("""
                INSERT INTO product_embeddings (product_id, embedding)
                VALUES (%s, %s);
            """, (product_id, vector))

        conn.commit()
        return {"status": "success", "product_id": product_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/api/scan")
async def scan_and_identify(file: UploadFile = File(...)):
    raw = await file.read()
    query_vector = extract_vector(raw)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                p.id, p.name, p.category, p.price_good, p.price_medium, p.price_low, p.stock,
                (1 - (pe.embedding <=> %s::vector)) AS similarity
            FROM product_embeddings pe
            JOIN products p ON p.id = pe.product_id
            ORDER BY pe.embedding <=> %s::vector ASC
            LIMIT 1;
        """, (query_vector, query_vector))

        row = cur.fetchone()
        if not row or row[7] < 0.72:  # Strict confidence threshold
            return {"matched": False, "message": "Product not found"}

        return {
            "matched": True,
            "product": {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "price_good": float(row[3]),
                "price_medium": float(row[4]),
                "price_low": float(row[5]),
                "stock": row[6],
                "confidence": round(float(row[7]) * 100, 1)
            }
        }
    finally:
        cur.close()
        conn.close()

@app.post("/api/products/{product_id}/sell")
async def record_sale(product_id: int):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM sell_product_item(%s);", (product_id,))
        new_stock = cur.fetchone()[0]
        conn.commit()
        return {"status": "sold", "new_stock": new_stock}
    finally:
        cur.close()
        conn.close()