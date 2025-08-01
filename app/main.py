import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import FastAPI app
from app.api.routes import app

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))

    # Run the application
    uvicorn.run(
        "app.api.routes:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )