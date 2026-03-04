// ============================================================
//  CHART EYE — Trading Coach BAML Schema
//  Senior Quant / 20-year Price Action veteran perspective
// ============================================================

// ── Candlestick pattern with confluence validation ───────────
class CandlePattern {
  type     string  @description("e.g. Hammer, Shooting Star, Bullish Engulfing, Bearish Engulfing, Doji, Marubozu, Morning Star, Evening Star, Spinning Top")
  location string  @description("Position on chart: bottom-of-range, top-of-range, mid-range, at-support, at-resistance")
  strength "Strong" | "Moderate" | "Weak"
  is_valid bool    @description("True ONLY if the pattern has structural confluence. A Hammer at mid-range is FALSE. A Hammer at the bottom of a 20-candle downmove is TRUE.")
  context  string  @description("1-2 sentences: why valid, or why discounted. Be specific.")
}

// ── Smart Money Concepts: 4-phase market model ───────────────
enum MarketPhase {
  Accumulation   @description("Smart money quietly building longs. Low volatility, range-bound. Springs and shakeouts common.")
  Manipulation   @description("Stop hunts and liquidity grabs. Price pierces obvious S/R then snaps back.")
  Distribution   @description("Smart money offloading into retail buying. Volatility expanding at highs. Bearish divergence.")
  Expansion      @description("Trending impulse move. Large candles, clear directional bias, momentum.")
}

// ── Chart geometry ───────────────────────────────────────────
class ChartShape {
  pattern     string  @description("Double Top, Double Bottom, Head and Shoulders, Inverse H&S, Bull Flag, Bear Flag, Ascending/Descending Triangle, Wedge, Channel, Cup & Handle")
  status      "Forming" | "Confirmed" | "Broken"
  implication string  @description("What this pattern means for near-term price direction")
}

// ── Fair Value Gap / Liquidity Void ──────────────────────────
class FairValueGap {
  direction   "Bullish" | "Bearish"  @description("Bullish FVG = price likely to fill up; Bearish FVG = price likely to fill down")
  description string                 @description("Location description, approximate price level if legible, and gap size (small/medium/large)")
}

// ── Zoom state assessment ────────────────────────────────────
class ZoomInstruction {
  state              "TooZoomedIn" | "Optimal" | "TooZoomedOut"
  estimated_candles  int     @description("Rough estimate of visible candles on chart")
  coaching_message   string  @description("Explain WHY the current zoom level is a problem or is good. Reference the candle count.")
  call_to_action     string  @description("Specific instruction: e.g. 'Switch to the 4H chart to see the supply zone you are currently hitting from below' or 'Zoom to 15m to confirm the entry trigger candle'")
}

// ── Master coaching output ───────────────────────────────────
class TradingCoachOutput {
  candle_patterns   CandlePattern[]   @description("ALL identifiable patterns including discounted ones — show your work")
  market_phase      MarketPhase
  phase_explanation string            @description("2-3 sentences of hard evidence for the market phase call")
  chart_shapes      ChartShape[]      @description("Geometric patterns visible, or empty array if none")
  fair_value_gaps   FairValueGap[]    @description("FVGs and liquidity voids identified, or empty array if none")
  zoom_assessment   ZoomInstruction
  coach_commentary  string            @description("Your main coaching insight. Use veteran trader language. Be direct, tactical, and specific. No fluff.")
  call_to_action    string            @description("The single most important thing the trader should do RIGHT NOW. One sentence.")
  bias              "Bullish" | "Bearish" | "Neutral"
  confidence        "High" | "Medium" | "Low"
}

// ── Main coaching function ───────────────────────────────────
function AnalyzeMarketContext(chart: image, query: string) -> TradingCoachOutput {
  client AzureGPT41
  prompt #"
    You are a 20-year veteran day trader and trading coach. You specialize in Price Action, Smart Money Concepts (SMC), Order Flow, and Market Microstructure. You speak with authority and precision — no vague generalities.

    A trader asks: "{{ query }}"

    Analyze the chart screenshot below using your expert eye:

    {{ chart }}

    Work through this analysis chain-of-thought:

    ═══ STEP 1: ZOOM VALIDATION ═══
    First, count approximately how many candles are visible:
    - Fewer than 30 candles → TooZoomedIn (you see detail but lose trend context — dangerous for entries)
    - 30 to 300 candles → Optimal (you can see both structure and candle confirmation)
    - More than 300 candles → TooZoomedOut (you see the trend but can't read entry triggers)
    Be honest about the count. A bad zoom kills good setups.

    ═══ STEP 2: CANDLE ANALYSIS WITH CONFLUENCE RULES ═══
    Identify every notable candlestick pattern. Apply these strict rules:
    - A HAMMER is only valid at the BOTTOM of a downtrend or range — a Hammer at mid-range is noise
    - A SHOOTING STAR is only valid at the TOP of an uptrend or range
    - A DOJI at mid-range is just indecision noise — mark is_valid as false
    - An ENGULFING pattern requires the body to fully engulf the previous candle's body
    - A MARUBOZU (no wicks) shows conviction — note the direction
    - EVENING/MORNING STAR requires a 3-candle sequence at extremes
    List every pattern you see, valid or not. Show your reasoning for each.

    ═══ STEP 3: MARKET PHASE (SMART MONEY CONCEPTS) ═══
    Determine the current phase: Accumulation / Manipulation / Distribution / Expansion.
    Look for:
    - Fair Value Gaps (FVG): the gap between the wick of candle N and the wick of candle N+2 — price tends to return to fill these
    - Liquidity pools: equal highs, equal lows, obvious swing points where stops cluster
    - Order blocks: the last bearish candle before a bullish move (demand OB) or last bullish candle before bearish move (supply OB)
    Cite specific evidence from the chart.

    ═══ STEP 4: CHART GEOMETRY ═══
    Identify any forming or confirmed geometric patterns. Be conservative — only call what you can clearly see.

    ═══ STEP 5: COACHING COMMENTARY ═══
    Synthesize everything into your main verdict. Use the language of an experienced trader:
    - "Price is hunting for liquidity above those equal highs before the real reversal"
    - "Smart money has already distributed — retail is about to buy the top"
    - "Stop-run incoming on those obvious swing lows — expect a wick and reversal"
    - "Sellers are exhausted. This consolidation is a spring. Watch for a break of structure to the upside."
    - "We're in a premium zone hitting a supply OB. Wait for confirmation before going long."
    Be direct. Be specific. Reference what you actually see on the chart.

    {{ ctx.output_format }}
  "#
}
