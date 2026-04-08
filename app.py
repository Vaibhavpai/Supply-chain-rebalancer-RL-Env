from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Supply Chain Inventory Rebalancer is Running!", "message": "Ready for OpenEnv evaluation."}