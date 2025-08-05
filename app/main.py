import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT"))

    # Run the application
    uvicorn.run(
        "app.api.routes:app",
        host=os.getenv("HOST"),
        port=port,
        reload=True
    )
