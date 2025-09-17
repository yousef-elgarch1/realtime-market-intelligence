from fastapi import FastAPI
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Market Intelligence API",
    description="BCG X Real-time Market Intelligence Platform",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Market Intelligence Platform API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "kafka": "connected",
        "redis": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)