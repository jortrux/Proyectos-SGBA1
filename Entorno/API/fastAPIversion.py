from fastapi import FastAPI

app = FastAPI(version="v0.1.0.0")

@app.get("/v0/version")
async def version():
    print({"version": app.version})
    return {"version": app.version}