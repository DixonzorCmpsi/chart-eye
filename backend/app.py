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
class ConversationMessage(BaseModel):
    role: str  # 'user' or 'ai'
    query: str
    verdict: str = ""
    reasoning: str = ""

class InferenceRequest(BaseModel):
    image: str
    query: str

class AnalyzeRequest(BaseModel):
    image:    str
    query:    str
    iteration: int = 0
    context:  str = ""
    history:  list[ConversationMessage] = []  # Conversation memory


# ============================================================
#  CONVERSATION MEMORY (in-memory store)
# ============================================================
conversation_history: list[ConversationMessage] = []

def add_to_history(role: str, query: str, verdict: str = "", reasoning: str = ""):
    """Add a message to conversation history"""
    msg = ConversationMessage(role=role, query=query, verdict=verdict, reasoning=reasoning)
    conversation_history.append(msg)
    logger.info(f"[history] Added {role} message: {query[:50]}")

def get_history_context() -> str:
    """Build context string from recent conversation"""
    if not conversation_history:
        return ""
    
    recent = conversation_history[-5:]  # Last 5 messages
    history_text = "RECENT CONVERSATION CONTEXT:\n"
    
    for msg in recent:
        if msg.role == 'user':
            history_text += f"- You asked: {msg.query}\n"
        else:
            history_text += f"- Chart Eye said: {msg.verdict}: {msg.reasoning[:100]}...\n"
    
    return history_text + "\n"

def clear_history():
    """Clear conversation history (for testing or fresh start)"""
    global conversation_history
    conversation_history = []
    logger.info("[history] Conversation history cleared")


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

{history_context}User asks: "{query}"

IMPORTANT CONTEXT NOTES:
- Remember previous trades and levels discussed in this conversation
- Look for patterns and confirmations from prior analysis
- If user asks about "the bounce" or "that level," they're referring to recent context - use it to inform your analysis
- Build on previous verdicts - don't contradict them unless price action clearly changes
- Reference what was previously identified (support at X, resistance at Y, etc.) to show continuity

INTERNAL REASONING (Think through this but don't show it):

STEP 1: OBSERVE WHAT YOU SEE
- Chart style/timeframe clues? (candle size, pattern frequency, zoom level)
- Current price action? (up/down/consolidating? strong/weak?)
- What are the LAST 3-5 candles telling you?
- Are there clear support/resistance levels?
- How does this relate to what we discussed before?

STEP 2: CHECK FOR LIQUIDITY SWEEPS
Liquidity sweeps = wicks that breach recent support/resistance then reverse sharply:
- Look for WICKS that extend 1.5x+ the candle body
- Do wicks penetrate recent support/resistance by 0.5-2%?
- Does the candle close 50%+ AWAY from the wick (showing strong reversal)?
- Are 1-3 follow-through candles confirming the reversal direction?
- This exhausts trapped stops and shows reversal conviction
- POWERFUL SIGNAL if sweep happens at confluence (previous high/low + current support/resistance)

STEP 3: IDENTIFY REVERSALS & STRUCTURE BREAKS
Reversal indicators:
- CANDLESTICK PATTERNS: Engulfing (large opposite body engulfs small), Pinbar (long wick + small body), Hammer/Shooting Star
- STRUCTURE BREAK: Uptrend was making higher lows → now makes LOWER low = bearish. Downtrend was making lower highs → now makes HIGHER high = bullish
- VOLUME SHIFT: Reversal candles on HIGHER volume than trend candles = confirms strength
- FOLLOW-THROUGH: 2-3 candles after pattern continue in new direction (NOT a false reversal)
- REJECTED WICKS: Long wicks in opposite direction of reversal show extreme rejection at key level

STEP 4: ANALYZE MOMENTUM & CONFLUENCE
- Is the current move strong or weak?
- Are candles getting bigger or smaller?
- Wick behavior - are rejections happening?
- KEY QUESTION: Do you see BOTH a liquidity sweep AND a reversal candle/pattern at the same level? = HIGHEST CONVICTION
- Does price have confluence (sweep + reversal pattern + structure break + volume + previous swing zone)? = Trade Setup Ready

STEP 5: DECIDE ON YOUR TRADE ACTION
Based on sweeps, reversals, and confluence, choose ONE of:
- BUY LONG (if liquidity sweep at support + bullish reversal candle + structure break (lower low reversed) + volume increase + follow-through)
- SELL SHORT (if liquidity sweep at resistance + bearish reversal candle + structure break (higher high reversed) + volume increase + follow-through)
- WAIT (if no clear sweep/reversal, no structure break, or consolidating without confluence)

STEP 6: INFORMATION GAPS
Ask for wider view, zoom in, or different context if needed?

---

OUTPUT FORMAT REQUIRED - EXACTLY THIS STRUCTURE:

**VERDICT: [BUY LONG / SELL SHORT / WAIT] | Confidence: [HIGH / MEDIUM / LOW]**
[2-3 sentences of reasoning. Include: what price action confirms this? (sweep/reversal) where's the entry/stop? what's the target? what could invalidate?]

---

Example outputs:

✓ "**VERDICT: BUY LONG | Confidence: HIGH**
Price swept below support taking stops, then reversed hard with strong follow-through—textbook liquidity grab. Entry: above the reversal candle, stop below the sweep wick at 6,750, target resistance at 6,795. Invalidated if price closes back below support."

✓ "**VERDICT: SELL SHORT | Confidence: HIGH**
Structure broke higher into resistance, formed a pinbar rejection candle, and swept above previous high grabbing longs—now reversing sharply. I'd enter on this reversal, stop above the sweep wick, targeting the support below. If support holds and rallies again, this setup fails."

✓ "**VERDICT: WAIT | Confidence: MEDIUM**
No clear liquidity sweep or reversal pattern yet—candles are indecisive with small bodies. I'd watch for either a clean sweep at the current zone or a structure break. Can you show me the last 10 candles to see if there's an emerging higher low or lower high?"

✓ "**VERDICT: BUY LONG | Confidence: MEDIUM**
Downtrend was making lower highs, just broke the structure with a higher high + engulfing reversal candle at support. Sweep looks partial—need to confirm follow-through. Entry: above this candle, stop below support, target the resistance above. If it fails to hold above the break, invalidated."

Bad outputs (don't do these):
✗ "Price might bounce" (no verdict)
✗ "I see candles" (not analytical)
✗ No mention of sweeps, reversals, or structure breaks
✗ No entry/stop/target specified
✗ "Here are 5 reasons..." (too long)"""

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
    # Get conversation history context
    history_context = get_history_context()
    
    # Format prompt with history
    prompt = COACH_PROMPT.format(history_context=history_context, query=query)
    
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
    
    # Add to conversation history
    add_to_history(role="ai", query=query, verdict=verdict_type, reasoning=coaching_advice)
    
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
        # Add user query to history on first iteration
        if request.iteration == 0:
            add_to_history(role="user", query=request.query)
        
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
