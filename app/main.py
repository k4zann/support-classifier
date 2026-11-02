from fastapi import FastAPI

app = FastAPI(title="Support classifier")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
