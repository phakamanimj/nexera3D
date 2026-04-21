"""
NexEra AI-Powered 3D Asset Pipeline — Backend
FastAPI server using Groq (free) for AI + Sketchfab for 3D models.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import httpx
import base64
import json
import os
import re
from pathlib import Path

app = FastAPI(title="NexEra 3D Asset Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SKETCHFAB_API_KEY = os.environ.get("SKETCHFAB_API_KEY", "")
SKETCHFAB_SEARCH_URL = "https://api.sketchfab.com/v3/models"

# Fallback GLB models if Sketchfab fails
FALLBACK_MODELS = {
    "hard hat":         "https://vazxmixjsiawhamofees.supabase.co/storage/v1/object/public/models/hard-hat/model.gltf",
    "fire extinguisher":"https://vazxmixjsiawhamofees.supabase.co/storage/v1/object/public/models/fire-extinguisher/model.gltf",
    "laptop":           "https://vazxmixjsiawhamofees.supabase.co/storage/v1/object/public/models/macbook/model.gltf",
    "phone":            "https://vazxmixjsiawhamofees.supabase.co/storage/v1/object/public/models/iphone-x/model.gltf",
    "chair":            "https://vazxmixjsiawhamofees.supabase.co/storage/v1/object/public/models/chair/model.gltf",
    "default":          "https://modelviewer.dev/shared-assets/models/Astronaut.glb",
}


async def call_groq(user_message: str) -> str:
    """Call Groq AI and return response text."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant for NexEra training platform. Always respond with valid JSON only. No markdown, no code fences, no extra text whatsoever."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.3,
        "max_tokens": 600
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            GROQ_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Groq API error: {response.text}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Could not parse Groq response.")


def build_prompt(description: str = "") -> str:
    """Build the prompt sent to Groq."""
    object_hint = f'The user described: "{description}"' if description else "Identify the main object in the image."
    return f"""{object_hint}

Return ONLY this JSON with no other text:
{{
  "object_name": "simple 1-3 word name of the object e.g. basketball, fire extinguisher, laptop",
  "educational_summary": "2-3 sentences: what it is, its use, and one safety tip.",
  "key_facts": ["fact 1", "fact 2", "fact 3"],
  "category": "one of: safety_equipment, tools, electronics, furniture, sports, objects, general",
  "training_context": "One sentence on how this relates to workplace training."
}}"""


def parse_ai_response(raw: str) -> dict:
    """Clean and parse AI JSON response."""
    raw = raw.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw.strip())


POLY_PIZZA_URL = "https://api.poly.pizza/v1/models"

async def search_model(query: str) -> dict:
    """
    Search Poly Pizza for a free 3D model.
    Returns a direct GLB URL — no API key needed.
    """
    try:
        print(f"Searching Poly Pizza for: {query}")
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                POLY_PIZZA_URL,
                params={"q": query, "limit": 1},
                headers={"X-Auth-Token": "1234"}
            )
        print(f"Poly Pizza status: {response.status_code}")
        print(f"Poly Pizza response: {response.text[:300]}")

        if response.status_code == 200:
            data = response.json()
            results = data.get("data", [])
            if results:
                model = results[0]
                glb_url = model.get("Download", {}).get("url", "")
                name = model.get("Title", query)
                if glb_url:
                    print(f"Found: {name} -> {glb_url}")
                    return {"found": True, "url": glb_url, "name": name, "type": "gltf"}
    except Exception as e:
        print(f"Poly Pizza error: {e}")

    return {"found": False}


def get_fallback_model(object_name: str) -> str:
    """Get a fallback GLB model URL based on object name."""
    name_lower = object_name.lower()
    for key, url in FALLBACK_MODELS.items():
        if key in name_lower:
            return url
    return FALLBACK_MODELS["default"]


def build_response(ai_result: dict, model_url: str, fallback_url: str,
                   model_key: str, viewer_type: str, input_type: str, query: str) -> dict:
    """Build the standard JSON response sent to the frontend."""
    return {
        "success": True,
        "input_type": input_type,
        "query": query,
        "object_name": ai_result.get("object_name", query),
        "educational_summary": ai_result.get("educational_summary", ""),
        "key_facts": ai_result.get("key_facts", []),
        "category": ai_result.get("category", "general"),
        "training_context": ai_result.get("training_context", ""),
        "model_url": model_url,
        "fallback_url": fallback_url,
        "model_key": model_key,
        "viewer_type": viewer_type
    }


async def get_model(object_name: str) -> tuple:
    result = await search_model(object_name)
    if result["found"]:
        return (
            result["url"],
            "https://modelviewer.dev/shared-assets/models/Astronaut.glb",
            result["name"],
            "gltf"
        )
    else:
        fallback = get_fallback_model(object_name)
        return (fallback, fallback, object_name, "gltf")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(static_dir / "index.html"))


@app.post("/api/analyze/text")
async def analyze_text(description: str = Form(...)):
    if not description or len(description.strip()) < 2:
        raise HTTPException(status_code=400, detail="Description too short.")

    try:
        # Step 1: Ask Groq to identify the object
        prompt = build_prompt(description=description.strip())
        raw = await call_groq(prompt)
        ai_result = parse_ai_response(raw)
        object_name = ai_result.get("object_name", description)
        print(f"AI identified: {object_name}")

        # Step 2: Search Sketchfab for the 3D model
        model_url, fallback_url, model_key, viewer_type = await get_model(object_name)
        print(f"Model: {model_url} ({viewer_type})")

        return JSONResponse(build_response(
            ai_result, model_url, fallback_url,
            model_key, viewer_type, "text", description
        ))

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Try again.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type.")

    image_data = await file.read()
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    try:
        b64_image = base64.standard_b64encode(image_data).decode("utf-8")

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{file.content_type};base64,{b64_image}"}
                        },
                        {
                            "type": "text",
                            "text": build_prompt()
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GROQ_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
            )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Groq Vision error: {response.text}")

        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        ai_result = parse_ai_response(raw)
        object_name = ai_result.get("object_name", "object")
        print(f"Image AI identified: {object_name}")

        model_url, fallback_url, model_key, viewer_type = await get_model(object_name)

        return JSONResponse(build_response(
            ai_result, model_url, fallback_url,
            model_key, viewer_type, "image", object_name
        ))

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Try again.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def list_models():
    return {"models": list(FALLBACK_MODELS.keys())}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai": "Groq llama-3.1-8b",
        "models": "Sketchfab",
        "groq_key_set": bool(GROQ_API_KEY),
        "sketchfab_key_set": bool(SKETCHFAB_API_KEY)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)