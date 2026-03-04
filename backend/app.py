import json
import logging
import base64
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# BAML-generated types for structured output validation
from baml_client.types import TradingCoachOutput

load_dotenv()

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chart-eye-backend")

# ── Azure OpenAI — DefaultAzureCredential (no key required) ──
ENDPOINT        = os.environ["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
API_VERSION     = os.environ["AZURE_OPENAI_VERSION"]

HOST = "127.0.0.1"
PORT = 8000

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)
openai_client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)
logger.info(f"Azure OpenAI ready (DefaultAzureCredential) — deployment: {DEPLOYMENT_NAME}")

# ── FastAPI ───────────────────────────────────────────────────
app = FastAPI(title="Chart Eye AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  SCHEMAS
# ============================================================
class InferenceRequest(BaseModel):
    image: str
    query: str

class AnalyzeRequest(BaseModel):
    image:     str
    query:     str
    iteration: int = 0
    context:   str = ""


# ============================================================
#  REGION MAPPING
# ============================================================
REGION_NAMES = {
    "FULL", "TOP_HALF", "BOTTOM_HALF", "LEFT_HALF", "RIGHT_HALF",
    "TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"
}

def region_to_coords(name: str, w: int, h: int) -> dict:
    hw, hh = w // 2, h // 2
    table = {
        "TOP_HALF":     {"x": 0,  "y": 0,  "w": w,  "h": hh},
        "BOTTOM_HALF":  {"x": 0,  "y": hh, "w": w,  "h": hh},
        "LEFT_HALF":    {"x": 0,  "y": 0,  "w": hw, "h": h},
        "RIGHT_HALF":   {"x": hw, "y": 0,  "w": hw, "h": h},
        "TOP_LEFT":     {"x": 0,  "y": 0,  "w": hw, "h": hh},
        "TOP_RIGHT":    {"x": hw, "y": 0,  "w": hw, "h": hh},
        "BOTTOM_LEFT":  {"x": 0,  "y": hh, "w": hw, "h": hh},
        "BOTTOM_RIGHT": {"x": hw, "y": hh, "w": hw, "h": hh},
    }
    return table.get(name, {"x": 0, "y": 0, "w": w, "h": h})

def image_size_from_b64(b64: str) -> tuple[int, int]:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    return img.size  # (width, height)


# ============================================================
#  REGION DETECTION  — fast single-word call
# ============================================================
REGION_PROMPT = (
    "You are analyzing a trading platform screenshot. "
    "Identify where the main price chart/candlestick graph is located.\n\n"
    "MUST respond with EXACTLY ONE word from this list:\n"
    "FULL | TOP_HALF | BOTTOM_HALF | LEFT_HALF | RIGHT_HALF | "
    "TOP_LEFT | TOP_RIGHT | BOTTOM_LEFT | BOTTOM_RIGHT\n\n"
    "Do not include any other text. Answer now:"
)

def detect_chart_region(base64_image: str) -> str:
    response = openai_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}},
                {"type": "text",      "text": REGION_PROMPT}
            ]
        }],
        max_tokens=12
    )
    raw = (response.choices[0].message.content or "").strip()
    print(f"[detect] raw: {repr(raw)}", flush=True)
    logger.info(f"[region-detect] raw: {repr(raw)}")
    for token in raw.upper().split():
        clean = token.strip(".,!?\"':-;/\\()")
        if clean in REGION_NAMES:
            print(f"[detect] found region: {clean}", flush=True)
            return clean
    logger.warning(f"[region-detect] no valid region in '{raw}', defaulting to FULL")
    print(f"[detect] defaulting to FULL", flush=True)
    return "FULL"


# ============================================================
#  COACHING ANALYSIS — structured TradingCoachOutput
#  Uses openai_client (DefaultAzureCredential) + JSON mode.
#  BAML handles schema definition and pydantic validation.
# ============================================================
COACH_PROMPT = """You are an expert trading analyst with deep reasoning skills. Your job is to analyze the chart and give a CLEAR TRADE VERDICT with reasoning.

User asks: "{query}"

INTERNAL REASONING (Think through this but don't show it):

STEP 1: OBSERVE WHAT YOU SEE
- Chart style/timeframe clues? (candle size, pattern frequency, zoom level)
- Current price action? (up/down/consolidating? strong/weak?)
- What are the LAST 3-5 candles telling you?
- Are there clear support/resistance levels?

STEP 2: IDENTIFY KEY ZONES
- Where is price relative to major support/resistance?
- Is there a trend or is price in a range?
- Where would buyers step in? Where would sellers defend?
- Any confluences (multiple levels at same area)?

STEP 3: ANALYZE MOMENTUM & STRUCTURE
- Is the current move strong or weak?
- Are candles getting bigger or smaller?
- Wick behavior - are rejections happening?
- Any emerging patterns (break, test, reversal)?

STEP 4: DECIDE ON YOUR TRADE ACTION
Choose ONE of:
- BUY LONG (if support is holding + momentum looks bullish)
- SELL SHORT (if resistance is holding + momentum looks bearish)
- WAIT (if unclear direction or consolidating)

STEP 5: INFORMATION GAPS
Ask for wider view, zoom in, or different context if needed?

---

OUTPUT FORMAT REQUIRED - EXACTLY THIS STRUCTURE:

**VERDICT: [BUY LONG / SELL SHORT / WAIT] | Confidence: [HIGH / MEDIUM / LOW]**
[2-3 sentences of reasoning. Include: what price action confirms this? where's the entry/stop? what's the target? what could invalidate?]

---

Example outputs:

✓ "**VERDICT: BUY LONG | Confidence: HIGH**
Support held strong with increasing buy volume. I'd enter long here, stop loss below support at 6,750, targeting resistance at 6,795. If price breaks below support on volume, this setup is invalidated."

✓ "**VERDICT: SELL SHORT | Confidence: MEDIUM**
Price rejected resistance twice with declining volume. I'd sell on next relief bounce back to the resistance line, stop above the recent high. Watch for breakdown below support to confirm acceleration lower."

✓ "**VERDICT: WAIT | Confidence: HIGH**
Price is consolidating with unclear direction—candles are small and indecisive. I'd wait for a clean break of either support or resistance on volume. Can you show a wider view to see the bigger trend context?"

Bad outputs (don't do these):
✗ "Price might bounce" (no verdict)
✗ "I see some candles" (not analytical)
✗ "Here are 5 reasons..." (too long)
✗ No confidence level given"""

def parse_verdict_response(raw: str) -> tuple[str, str, str, str]:
    """
    Parse the model's verdict response.
    Returns: (coaching_advice, verdict_type, confidence, call_to_action)
    
    Expected format:
    **VERDICT: BUY LONG | Confidence: HIGH**
    [reasoning text]
    """
    lines = raw.strip().split('\n')
    verdict_line = lines[0] if lines else ""
    reasoning = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
    
    # Extract verdict type and confidence
    verdict_type = "WAIT"
    confidence = "MEDIUM"
    
    if "BUY LONG" in verdict_line.upper():
        verdict_type = "BUY LONG"
    elif "SELL SHORT" in verdict_line.upper():
        verdict_type = "SELL SHORT"
    else:
        verdict_type = "WAIT"
    
    if "HIGH" in verdict_line.upper():
        confidence = "HIGH"
    elif "LOW" in verdict_line.upper():
        confidence = "LOW"
    else:
        confidence = "MEDIUM"
    
    # Dynamic call-to-action based on verdict
    cta_map = {
        "BUY LONG": "What position size are you considering? Any questions on entry or stop loss?",
        "SELL SHORT": "Are you comfortable with the risk/reward on this short? Want to adjust stops?",
        "WAIT": "Want me to analyze a different timeframe or zoomed-out view?"
    }
    call_to_action = cta_map.get(verdict_type, "What would you like to explore next?")
    
    # Clean up the coaching advice (remove markdown **VERDICT:** line)
    coaching_advice = reasoning if reasoning else raw.strip()
    
    return coaching_advice, verdict_type, confidence, call_to_action


def coach_analysis(base64_image: str, query: str) -> dict:
    prompt = COACH_PROMPT.format(query=query)
    
    if not base64_image or len(base64_image) < 100:
        logger.error(f"[coach] Invalid image: too short ({len(base64_image) if base64_image else 0} bytes)")
        return {
            "coach_commentary": "No valid chart image provided. Please upload a screenshot.",
            "call_to_action": "Share a chart screenshot so we can analyze it together.",
            "market_phase": "N/A", "bias": "N/A", "confidence": "Low"
        }

    try:
        response = openai_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=500
        )
    except Exception as api_err:
        logger.error(f"[coach] API call failed: {api_err}")
        return {
            "coach_commentary": f"Couldn't analyze the chart right now. Let's try again.",
            "call_to_action": "Try uploading the screenshot again.",
            "market_phase": "N/A", "bias": "N/A", "confidence": "Low"
        }

    raw: str = response.choices[0].message.content or ""
    logger.info(f"[coach] response: {raw[:500]!r}")

    # Parse the verdict response to extract structured data
    coaching_advice, verdict_type, confidence, call_to_action = parse_verdict_response(raw)
    
    return {
        "coach_commentary": coaching_advice,
        "call_to_action": call_to_action,
        "market_phase": "Analysis", 
        "bias": verdict_type, 
        "confidence": confidence
    }


# ============================================================
#  ENDPOINTS
# ============================================================
@app.get("/")
async def health_check():
    return {
        "status": "ready",
        "model":   DEPLOYMENT_NAME,
        "auth":    "DefaultAzureCredential",
        "baml":    "trading_coach v1"
    }


@app.post("/predict")
async def predict(request: InferenceRequest):
    """Single-shot legacy endpoint."""
    try:
        coaching = coach_analysis(request.image, request.query)
        return {"prediction": coaching["coach_commentary"]}
    except Exception as e:
        logger.error(f"[/predict] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Iterative crop-and-analyze endpoint.

    iteration 0: detect chart region (fast single-word call)
      → if FULL: run coaching analysis → return answer
      → else: return crop coordinates

    iteration 1+: run coaching analysis on cropped image → return answer
    """
    logger.info(f"[/analyze] iter={request.iteration} q={request.query[:50]!r} ctx={request.context!r}")

    try:
        if request.iteration == 0:
            try:
                region_name = detect_chart_region(request.image)
                logger.info(f"[/analyze] detected region: {region_name}")
            except Exception as region_err:
                logger.error(f"[/analyze] region detection failed: {region_err}", exc_info=True)
                region_name = "FULL"

            if region_name == "FULL":
                try:
                    coaching = coach_analysis(request.image, request.query)
                    if not isinstance(coaching, dict):
                        coaching = {
                            "candle_patterns": [], "chart_shapes": [], "fair_value_gaps": [],
                            "coach_commentary": "Analysis error: invalid response type",
                            "call_to_action": "", "phase_explanation": "",
                            "market_phase": "Expansion", "bias": "Neutral", "confidence": "Low",
                            "zoom_assessment": {"state": "Optimal", "estimated_candles": 0,
                                                "coaching_message": "", "call_to_action": ""}
                        }
                    prediction = coaching.get("coach_commentary", "Analysis unavailable")
                    logger.info(f"[/analyze] returning answer with coaching")
                    return {"action": "answer", "prediction": prediction, "coaching": coaching}
                except Exception as coach_err:
                    logger.error(f"[/analyze] coach_analysis failed: {coach_err}", exc_info=True)
                    return {"action": "answer", "prediction": f"Analysis error: {str(coach_err)[:100]}", 
                            "coaching": {"candle_patterns": [], "chart_shapes": [], "fair_value_gaps": [],
                                       "coach_commentary": f"Analysis failed: {str(coach_err)[:100]}",
                                       "call_to_action": "", "phase_explanation": "",
                                       "market_phase": "Expansion", "bias": "Neutral", "confidence": "Low",
                                       "zoom_assessment": {"state": "Optimal", "estimated_candles": 0,
                                                           "coaching_message": "", "call_to_action": ""}}}

            try:
                w, h = image_size_from_b64(request.image)
                coords = region_to_coords(region_name, w, h)
                logger.info(f"[/analyze] returning crop for region {region_name}")
                return {"action": "crop", "region": coords, "context": region_name}
            except Exception as crop_err:
                logger.error(f"[/analyze] crop preparation failed: {crop_err}", exc_info=True)
                return {"action": "crop", "region": {"x": 0, "y": 0, "w": 1, "h": 1}, "context": region_name}

        else:
            try:
                coaching = coach_analysis(request.image, request.query)
                if not isinstance(coaching, dict):
                    coaching = {
                        "candle_patterns": [], "chart_shapes": [], "fair_value_gaps": [],
                        "coach_commentary": "Analysis error: invalid response type",
                        "call_to_action": "", "phase_explanation": "",
                        "market_phase": "Expansion", "bias": "Neutral", "confidence": "Low",
                        "zoom_assessment": {"state": "Optimal", "estimated_candles": 0,
                                            "coaching_message": "", "call_to_action": ""}
                    }
                prediction = coaching.get("coach_commentary", "Analysis unavailable")
                logger.info(f"[/analyze] returning answer (iter>0) with coaching")
                return {"action": "answer", "prediction": prediction, "coaching": coaching}
            except Exception as coach_err:
                logger.error(f"[/analyze] coach_analysis failed (iter>0): {coach_err}", exc_info=True)
                return {"action": "answer", "prediction": f"Analysis error: {str(coach_err)[:100]}", 
                        "coaching": {"candle_patterns": [], "chart_shapes": [], "fair_value_gaps": [],
                                   "coach_commentary": f"Analysis failed: {str(coach_err)[:100]}",
                                   "call_to_action": "", "phase_explanation": "",
                                   "market_phase": "Expansion", "bias": "Neutral", "confidence": "Low",
                                   "zoom_assessment": {"state": "Optimal", "estimated_candles": 0,
                                                       "coaching_message": "", "call_to_action": ""}}}

    except Exception as outer_err:
        logger.error(f"[/analyze] OUTER ERROR (should not happen): {outer_err}", exc_info=True)
        return {"action": "error", "prediction": f"Unexpected error: {str(outer_err)[:100]}", 
                "coaching": {"candle_patterns": [], "chart_shapes": [], "fair_value_gaps": [],
                           "coach_commentary": f"Unexpected error: {str(outer_err)[:100]}",
                           "call_to_action": "", "phase_explanation": "",
                           "market_phase": "Expansion", "bias": "Neutral", "confidence": "Low",
                           "zoom_assessment": {"state": "Optimal", "estimated_candles": 0,
                                               "coaching_message": "", "call_to_action": ""}}}


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
