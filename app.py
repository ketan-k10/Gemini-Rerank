import json
import os
from dotenv import load_dotenv
load_dotenv()
import re
import io
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, Response, stream_with_context
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

app = Flask(__name__)

EEN_SEARCH_URL = "https://api.c001.eagleeyenetworks.com/api/v3.0/videoAnalyticEvents:deepSearch"
EEN_INCLUDES = (
    "data.een.objectDetection.v1,"
    "data.een.vehicleAttributes.v1,"
    "data.een.personAttributes.v1,"
    "data.een.objectClassification.v1,"
    "data.een.fullFrameImageUrl.v1,"
    "data.een.croppedFrameImageUrl.v1,"
    "data.een.customLabels.v1,"
    "data.een.eevaAttributes.v1"
)
GEMINI_MODEL   = "gemini-3.1-flash-lite-preview"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MAX_DISPLAY_WIDTH = 640   # thumbnails only — Gemini always gets full resolution

# ── Prompt presets ────────────────────────────────────────────────────────────

_P1 = """You are a precision visual relevance assessor for a surveillance search system.
You will score {n} images against a search query using adaptive weighting based on the query type.

SEARCH QUERY: "{query}"

Images are labeled [Image 0] through [Image {n_minus_1}].
Each image has red bounding box(es) drawn around the detected object(s) that triggered the search result,
plus text metadata describing those objects' attributes.

════════════════════════════════════════
STEP 1 — CLASSIFY THE QUERY
════════════════════════════════════════
Before scoring any image, decide which query type applies:

  APPEARANCE QUERY — the query describes specific visual attributes of a person or object:
    Signals: clothing color, garment type, accessory, held item, physical appearance
    Examples: "person in blue hoodie", "woman holding cup", "man with red cap"
    → Apply APPEARANCE WEIGHTS (bounding box is the only visual evidence)

  SCENE QUERY — the query describes an activity, location, or contextual situation:
    Signals: actions, environments, interactions, spatial context
    Examples: "person near exit door", "people sitting at table", "crowded hallway"
    → Apply SCENE WEIGHTS (full image is primary evidence)

  MIXED QUERY — contains both appearance attributes AND scene/activity elements:
    → Apply APPEARANCE WEIGHTS (appearance attributes take precedence)

════════════════════════════════════════
APPEARANCE WEIGHTS  (bbox-only)
════════════════════════════════════════
For appearance queries the full scene image is IRRELEVANT — only the bounding box and metadata count.
Do NOT scan or use the background/scene for appearance queries under any circumstance.

  [A] Bounding Box Visual Match       — 75 pts max
      Inspect ONLY the content INSIDE the red bounding box.
      Do the person/object attributes match ALL descriptors in the query?
      — All attributes confirmed (correct subject, color, garment/item) → 65-75 pts
      — Minor uncertainty (lighting, angle, partial occlusion) → 45-62 pts
      — Subject correct but one key attribute wrong (wrong color OR wrong garment) → 15-30 pts
      — Multiple attributes wrong → 3-12 pts
      — No bounding box visible → 0 pts — image scores from Factor B only, max 25 pts total
      ★ COLOR IS A HARD FILTER: if the query specifies a color and the bounding box clearly
        shows a different color, Factor A is capped at 15 pts regardless of other matches.

  [B] Object Attribute Metadata       — 25 pts max
      Review the text metadata (class, clothing color, gender, held objects).
      — Explicitly confirms subject type AND all query attributes → 22-25 pts
      — Confirms subject type; attributes partially match → 10-18 pts
      — Generic metadata (e.g. just "person") → 4-8 pts
      — Contradicts a query attribute → 0 pts

  ✗ Full scene context: DO NOT award any points for what appears outside the bounding box.
    The bounding box is the ONLY visual evidence for appearance queries.

════════════════════════════════════════
SCENE WEIGHTS  (full-image-primary)
════════════════════════════════════════
  [A] Full Image Scene Match          — 55 pts max
      Look at the entire image.
      Does the scene show the activity, context, or situation described in the query?
      — Scene fully matches the described context → 48-55 pts
      — Scene partially matches (correct setting, activity slightly off) → 25-45 pts
      — Scene loosely related → 10-22 pts
      — Scene unrelated → 0 pts

  [B] Bounding Box Object Relevance   — 30 pts max
      Does the detected object inside the bounding box directly participate in the scene?
      — Object is a central element of the scene → 25-30 pts
      — Object is present but peripheral → 10-20 pts
      — Object unrelated to the described scene → 0 pts
      — No bounding box visible → 0 pts

  [C] Metadata Alignment              — 15 pts max
      Does the object metadata support the scene interpretation?
      — Strongly confirms scene context → 13-15 pts
      — Partial support → 5-10 pts
      — Contradicts scene → 0 pts

════════════════════════════════════════
CALIBRATION ANCHORS  (both query types)
════════════════════════════════════════
  90-100  Perfect match — all attributes or scene elements confirmed without doubt
  70-89   Strong match — key elements confirmed, minor visual uncertainty
  50-69   Good match — subject/scene correct, one secondary element slightly off
  30-49   Weak match — correct subject type but a key attribute or context is wrong
  10-29   Tangential — barely related, no meaningful attribute alignment
   0-9    Irrelevant — wrong subject, wrong scene, or image completely unrelated

RULES:
- Score every image. Assign 0 freely — most images might not match.
- A false positive is worse than a false negative — be strict.
- If no red bounding box is visible, rely on Factor B (appearance) or Factor B+C (scene) only.
- Do not inflate scores for "almost right" results.

════════════════════════════════════════
OUTPUT FORMAT — exactly {n} lines, sorted highest score first:
<index>:<score>

No explanations. No headers. No extra text. Only sorted index:score pairs.
════════════════════════════════════════""".strip()

_P2 = """You are a scene-aware visual relevance assessor for a surveillance search system.
Score {n} images (labeled [Image 0] through [Image {n_minus_1}]) for this query:

SEARCH QUERY: "{query}"

Each image shows a full surveillance frame. Red bounding boxes mark detected objects.
Text metadata describes those objects' attributes.

APPROACH: Evaluate the COMPLETE SCENE first — environment, activity, and context — then
check the bounding box for subject confirmation.

════════════════════════════════════════
SCORING CRITERIA
════════════════════════════════════════
[A] Full Scene Context — 60 pts max
    Examine the entire frame: setting, activity, number of people, spatial relationships.
    — Scene fully matches the query context → 50-60 pts
    — Scene partially matches (right setting, activity slightly off) → 28-48 pts
    — Scene loosely connected → 10-25 pts
    — Scene unrelated → 0 pts

[B] Bounding Box Subject Confirmation — 30 pts max
    Does the detected object inside the bbox match the query subject?
    — Perfect subject match (type + key attributes) → 25-30 pts
    — Subject type correct, minor attribute mismatch → 12-22 pts
    — Wrong subject type → 0-8 pts
    — No bounding box → 0 pts

[C] Object Metadata Support — 10 pts max
    Does the attribute text reinforce the scene interpretation?
    — Strongly confirms → 8-10 pts
    — Partial support → 3-6 pts
    — Contradicts → 0 pts

════════════════════════════════════════
CALIBRATION
════════════════════════════════════════
  90-100  Scene perfectly matches, subject confirmed
  70-89   Strong scene match, minor uncertainty
  50-69   Correct scene/subject, one detail off
  30-49   Scene or subject only partially relevant
  10-29   Weak scene connection
   0-9    Irrelevant or wrong scene entirely

RULES:
- Background, environment, and activity context are primary evidence here.
- Use lighting, time of day, architecture, and crowd density as scene signals.
- Assign 0 freely — most images will not match.

OUTPUT FORMAT — exactly {n} lines, sorted highest score first:
<index>:<score>
No explanations. No headers. No extra text. Only sorted index:score pairs.
════════════════════════════════════════""".strip()

_P3 = """You are a strict color-and-appearance matcher for surveillance footage.
Score {n} images (labeled [Image 0] through [Image {n_minus_1}]) for this query:

SEARCH QUERY: "{query}"

Red bounding boxes mark detected objects. Inspect ONLY inside the bounding box.

════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════
1. ONLY inspect the content inside the red bounding box — ignore everything outside.
2. COLOR BINARY GATE: if the query specifies a color and the bounding box clearly shows
   a DIFFERENT color, the image scores 0. No exceptions.
3. Garment or object TYPE must match exactly — "similar" types do not count.
4. If there is no visible bounding box, max possible score is 15.

════════════════════════════════════════
SCORING (inside bbox only)
════════════════════════════════════════
[A] Color Match — 50 pts max
    Does the dominant color inside the bbox exactly match the query color?
    — Exact or near-exact color match → 42-50 pts
    — Possible match (heavy shadow, partial occlusion, unusual lighting) → 22-38 pts
    — Clearly different color → 0 pts AND total score capped at 0

[B] Garment / Object Type — 30 pts max
    Does the clothing type or object type match the query exactly?
    — Exact match (e.g. "hoodie" = hoodie) → 25-30 pts
    — Very close type (e.g. "jacket" vs "hoodie") → 12-20 pts
    — Wrong type entirely → 0 pts

[C] Subject & Metadata Confirmation — 20 pts max
    Is the person/object type correct? Does metadata confirm query attributes?
    — Subject confirmed + metadata matches all attributes → 16-20 pts
    — Subject confirmed, metadata partially matches → 8-14 pts
    — Subject type wrong → 0 pts
    — Metadata explicitly contradicts → 0 pts

════════════════════════════════════════
CALIBRATION
════════════════════════════════════════
  90-100  Color exact, type exact, subject confirmed — all attributes verified
  70-89   Color and type confirmed, minor visual uncertainty only
  50-69   Color or type slightly uncertain but plausible
  30-49   One attribute clearly wrong, others match
   0-29   Color wrong OR subject type wrong

RULES:
- Be strict. A false positive is far worse than a false negative.
- Shadows and compression artifacts lower color certainty — reflect this in scoring.
- Never exceed 15 pts if no bounding box is visible.

OUTPUT FORMAT — exactly {n} lines, sorted highest score first:
<index>:<score>
No explanations. No headers. No extra text. Only sorted index:score pairs.
════════════════════════════════════════""".strip()

_P4 = """You are a balanced visual relevance assessor for a surveillance search system.
Score {n} images (labeled [Image 0] through [Image {n_minus_1}]) for this query:

SEARCH QUERY: "{query}"

Each image shows a full surveillance frame with red bounding boxes around detected objects,
plus text metadata. Weigh visual evidence, scene context, and metadata roughly equally.

════════════════════════════════════════
SCORING — 100 POINTS TOTAL
════════════════════════════════════════
[A] Bounding Box Visual Match — 40 pts max
    Examine inside the red bounding box.
    — Subject type and key attributes match the query → 32-40 pts
    — Subject correct, one attribute off → 18-30 pts
    — Subject wrong type → 0-10 pts
    — No bounding box visible → 0 pts

[B] Scene / Environment Context — 35 pts max
    Look at the full frame: does the background, activity, or setting support the query?
    — Scene strongly supports the query → 28-35 pts
    — Scene somewhat relevant → 14-26 pts
    — Scene unrelated to query → 0-10 pts

[C] Object Attribute Metadata — 25 pts max
    Does the text metadata (class, color, gender, held objects) confirm the query?
    — All key attributes confirmed → 20-25 pts
    — Partial confirmation → 10-18 pts
    — Contradiction → 0 pts

════════════════════════════════════════
CALIBRATION
════════════════════════════════════════
  85-100  All three factors confirm the query strongly
  65-84   Two factors confirm clearly, third is partial
  45-64   Mixed signals — some factors match, others uncertain
  25-44   Weak overall match — minimal relevant evidence
   0-24   Not relevant across most or all factors

RULES:
- Score all images. Most images will score below 30.
- Do not over-reward "almost right" results.
- If bounding box is absent, weight B and C more heavily.
- Be fair but realistic about what counts as a match.

OUTPUT FORMAT — exactly {n} lines, sorted highest score first:
<index>:<score>
No explanations. No headers. No extra text. Only sorted index:score pairs.
════════════════════════════════════════""".strip()

DEFAULT_PROMPT = _P1

PROMPT_PRESETS = [
    {"id": "bbox_priority", "label": "Prompt 1 — BBox Priority (Adaptive)",  "text": _P1},
    {"id": "scene_context", "label": "Prompt 2 — Scene Context (Full Frame)", "text": _P2},
    {"id": "strict_color",  "label": "Prompt 3 — Strict Color Match",         "text": _P3},
    {"id": "balanced",      "label": "Prompt 4 — Balanced Holistic",          "text": _P4},
]


def sse_event(data):
    return f"data: {json.dumps(data)}\n\n"


def extract_event_info(result):
    match_ids = set(result.get("matchObjectIds", []))
    detections = {}
    attributes = {}
    image_url = None
    for item in result.get("data", []):
        t = item.get("type")
        oid = item.get("objectId")
        if t == "een.fullFrameImageUrl.v1":
            image_url = item.get("httpsUrl")
        elif t == "een.objectDetection.v1" and oid:
            detections[oid] = item.get("boundingBox", [])
        elif t in ("een.personAttributes.v1", "een.objectClassification.v1") and oid:
            if oid not in attributes:
                attributes[oid] = {}
            attributes[oid].update({
                k: v for k, v in item.items()
                if k not in ("type", "creatorId", "objectId")
            })
    matched_boxes = {oid: detections[oid] for oid in match_ids if oid in detections}
    matched_attrs = {oid: attributes.get(oid, {}) for oid in match_ids}
    return image_url, matched_boxes, matched_attrs


def _resize_for_display(img):
    if img.width > MAX_DISPLAY_WIDTH:
        ratio = MAX_DISPLAY_WIDTH / img.width
        return img.resize((MAX_DISPLAY_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    return img


def _to_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def process_result(idx, result, token):
    image_url, matched_boxes, matched_attrs = extract_event_info(result)
    if not image_url:
        return None

    resp = requests.get(
        image_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    resp.raise_for_status()

    # Full-resolution original — never downscaled
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    w, h = img.size

    # Annotate at full resolution — line width scales with image size
    bbox_line_width = max(3, int(w / 320))
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for bbox in matched_boxes.values():
        if len(bbox) == 4:
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=bbox_line_width)

    # Full-res annotated → Gemini (high quality, no resize)
    annotated_gemini_bytes = _to_jpeg(annotated, quality=95)

    # Display copies (resized, smaller file)
    display_annotated = _resize_for_display(annotated)

    parts = []
    for attrs in matched_attrs.values():
        desc = ", ".join(f"{k}: {v}" for k, v in attrs.items() if k != "class")
        cls = attrs.get("class", "object")
        parts.append(f"{cls} ({desc})" if desc else cls)
    attr_summary = "; ".join(parts) if parts else "no attributes"

    return {
        "idx": idx,
        "display_annotated": base64.b64encode(_to_jpeg(display_annotated, quality=82)).decode(),
        "annotated_gemini":  base64.b64encode(annotated_gemini_bytes).decode(),
        "attr_summary": attr_summary,
        "result_id": result.get("id", ""),
        "timestamp": result.get("startTimestamp", ""),
        "actor": result.get("actorName", ""),
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_prompt=DEFAULT_PROMPT,
        prompt_presets=PROMPT_PRESETS,
    )


@app.route("/search", methods=["POST"])
def search():
    req = request.json or {}
    token        = req.get("token", "").strip()
    prompt_tmpl  = req.get("prompt", DEFAULT_PROMPT)
    query        = req.get("query", "").strip()
    disable_spell_check = req.get("disable_spell_check", False)
    ts_gte = req.get("ts_gte", "")
    ts_lte = req.get("ts_lte", "")

    def generate():
        yield sse_event({"type": "progress", "step": 1, "msg": "Calling EEN DeepSearch API..."})

        try:
            api_resp = requests.post(
                EEN_SEARCH_URL,
                params={
                    "pageSize": 50,
                    "timestamp__gte": ts_gte,
                    "timestamp__lte": ts_lte,
                    "include": EEN_INCLUDES,
                    "sort": "+relevance",
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"query": query, "disableSpellCheck": disable_spell_check},
                timeout=30,
            )
            api_resp.raise_for_status()
            results = api_resp.json().get("results", [])[:50]
        except Exception as e:
            yield sse_event({"type": "error", "msg": f"EEN Search API failed: {e}"})
            return

        if not results:
            yield sse_event({"type": "error", "msg": "No results returned from EEN API."})
            return

        yield sse_event({
            "type": "progress", "step": 2,
            "msg": f"Got {len(results)} results. Downloading & annotating images in parallel...",
        })

        processed = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_result, i, r, token): i
                for i, r in enumerate(results)
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    item = future.result()
                    if item:
                        processed[item["idx"]] = item
                except Exception:
                    pass
                if done % 10 == 0 or done == len(futures):
                    yield sse_event({
                        "type": "progress", "step": 2,
                        "msg": f"Downloaded & annotated {done}/{len(futures)} images...",
                    })

        ordered = [processed[i] for i in range(len(results)) if i in processed]

        if not ordered:
            yield sse_event({"type": "error", "msg": "No images could be downloaded. Check your auth token."})
            return

        yield sse_event({
            "type": "progress", "step": 3,
            "msg": f"Prepared {len(ordered)} images. Sending to Gemini ({GEMINI_MODEL}) for reranking...",
        })

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            n = len(ordered)
            prompt = (
                prompt_tmpl
                .replace("{query}", query)
                .replace("{n}", str(n))
                .replace("{n_minus_1}", str(n - 1))
            )
            contents = [types.Part.from_text(text=prompt)]
            for i, item in enumerate(ordered):
                contents.append(types.Part.from_text(
                    text=f"[Image {i}] File: {i}.jpeg | Matched: {item['attr_summary']}"
                ))
                contents.append(types.Part.from_bytes(
                    data=base64.b64decode(item["annotated_gemini"]),
                    mime_type="image/jpeg",
                ))

            t0 = time.time()
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
            latency = round(time.time() - t0, 1)
            ranking_text = resp.text.strip()
        except Exception as e:
            yield sse_event({"type": "error", "msg": f"Gemini reranking failed: {e}"})
            return

        ranked = []
        seen = set()
        for m in re.finditer(r"(\d+)\s*:\s*(\d+)", ranking_text):
            idx, score = int(m.group(1)), int(m.group(2))
            if idx not in seen and 0 <= idx < n:
                seen.add(idx)
                ranked.append((idx, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for i in range(n):
            if i not in seen:
                ranked.append((i, 0))

        # Both columns show annotated images with bounding boxes
        original_out = [
            {
                "rank": i + 1,
                "image": item["display_annotated"],
                "attr_summary": item["attr_summary"],
                "timestamp": item["timestamp"],
                "actor": item["actor"],
            }
            for i, item in enumerate(ordered)
        ]

        reranked_out = [
            {
                "rank": rank,
                "original_rank": idx + 1,
                "score": score,
                "image": ordered[idx]["display_annotated"],
                "attr_summary": ordered[idx]["attr_summary"],
                "timestamp": ordered[idx]["timestamp"],
                "actor": ordered[idx]["actor"],
            }
            for rank, (idx, score) in enumerate(ranked[:50], 1)
        ]

        yield sse_event({
            "type": "done",
            "original": original_out,
            "reranked": reranked_out,
            "total": len(results),
            "latency": latency,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
