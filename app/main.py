import uvicorn
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    # Run the application
    uvicorn.run(
        "app.api.routes:app",
        host=host,
        port=port,
        reload=True
    )
