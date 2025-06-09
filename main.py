import os
import json
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Any, Set
import logging

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from dotenv import load_dotenv

import google.generativeai as genai 
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY") 
if not GEMINI_API_KEY:
     raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set in .env file")

AI_MODEL_GEMINI = "gemini-1.5-flash-latest"

CSV_FILE_PATH = '50_exercises_dataset (1).csv'
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Initialize Gemini Client ---
try:
    genai.configure(api_key=GEMINI_API_KEY) # ADDED

    gemini_model = genai.GenerativeModel(model_name=AI_MODEL_GEMINI) # Simpler initialization
    LOG.info(f"Google Gemini AI model '{AI_MODEL_GEMINI}' configured successfully.")
except Exception as e:
    LOG.error(f"Failed to configure Google Gemini AI: {e}")
    raise RuntimeError(f"Could not configure Gemini AI: {e}")


# --- Data Models (Pydantic for validation & docs) ---
class UserProfile(BaseModel):
    name: str
    age: int
    gender: str
    goal: str # e.g., muscle_gain, weight_loss, fitness
    experience: str = Field(..., pattern="^(beginner|intermediate|advanced)$")
    equipment: List[str]
    days_per_week: int = 3 # Requirement is 12 sessions, 3/week

class WorkoutPlanResponse(BaseModel):
     plan: List[Dict[str, Any]]
     ai_model_used: str
     notes: str

# --- FastAPI App ---
app = FastAPI(
    title="MyFit Mantra AI Workout Generator (Gemini Edition)",
     description="AI-Powered 12-session workout plan generator using Google Gemini API.",
     version="1.1.0"
)

# --- CORS Middleware (remains same) ---
origins = [
    "http://localhost", "http://localhost:8000", "http://127.0.0.1", "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

EXERCISE_DF = None

@app.on_event("startup")
def load_exercises():
    global EXERCISE_DF
    LOG.info(f"Loading exercises from {CSV_FILE_PATH}...")
    try:
        EXERCISE_DF = pd.read_csv(CSV_FILE_PATH)
        EXERCISE_DF = EXERCISE_DF.fillna('')
        EXERCISE_DF['equipment'] = EXERCISE_DF['equipment'].str.lower().str.strip()
        EXERCISE_DF['level'] = EXERCISE_DF['level'].str.lower().str.strip()
        EXERCISE_DF['type'] = EXERCISE_DF['type'].str.lower().str.strip()
        EXERCISE_DF['muscle_group'] = EXERCISE_DF['muscle_group'].str.lower().str.strip()
        LOG.info(f"Successfully loaded {len(EXERCISE_DF)} exercises.")
    except FileNotFoundError:
         LOG.error(f"ERROR: Exercise file not found at {CSV_FILE_PATH}")
         EXERCISE_DF = pd.DataFrame()
    except Exception as e:
        LOG.error(f"ERROR loading or parsing CSV: {e}")
        EXERCISE_DF = pd.DataFrame()

# --- Core Logic ---
def filter_exercises(profile: UserProfile) -> List[Dict]:
    # (This function remains the same as before)
    if EXERCISE_DF is None or EXERCISE_DF.empty:
         LOG.error("Exercise DataFrame is not loaded or is empty.")
         return []
    df = EXERCISE_DF.copy()
    available_equipment: Set[str] = set(e.lower().strip() for e in profile.equipment)
    available_equipment.add('bodyweight')
    available_equipment.add('none')
    if 'dumbbells' in available_equipment:
         available_equipment.add('dumbbell')
    allowed_levels: Set[str] = {profile.experience.lower(), 'all'}
    mask = (df['equipment'].isin(available_equipment)) & (df['level'].isin(allowed_levels))
    filtered_df = df[mask]
    LOG.info(f"Filtered exercises: {len(filtered_df)} available for level '{profile.experience}' with equipment {available_equipment}")
    return filtered_df.to_dict(orient='records')


def generate_dates(num_sessions: int = 12):
    # (This function remains the same as before)
    dates = []
    today = date.today()
    start_date = today + timedelta(days=1)
    if start_date.weekday() > 4 :
        start_date = start_date + timedelta(days=(7 - start_date.weekday()))
    day_offsets = [0, 2, 4]
    for i in range(num_sessions):
        week_number = i // 3
        day_in_schedule = i % 3
        current_session_base_date = start_date + timedelta(weeks=week_number)
        current_session_base_date_weekday = current_session_base_date.weekday()
        if day_offsets[day_in_schedule] < current_session_base_date_weekday:
             current_session_base_date += timedelta(days = 7 - current_session_base_date_weekday + day_offsets[day_in_schedule])
        else:
            current_session_base_date += timedelta(days = day_offsets[day_in_schedule] - current_session_base_date_weekday)
        if current_session_base_date <= today:
            current_session_base_date = today + timedelta(days=1 + day_offsets[day_in_schedule])
        if dates and current_session_base_date <= date.fromisoformat(dates[-1]):
            current_session_base_date = date.fromisoformat(dates[-1]) + timedelta(days=2 if day_in_schedule != 0 else 3)
        dates.append(current_session_base_date.isoformat())
    return dates

def construct_gemini_prompt(profile: UserProfile, exercise_list: List[Dict]) -> str:
    """Creates the prompt for the Gemini LLM. Gemini prefers a single, comprehensive prompt."""
    push_muscles = ["chest", "shoulders", "triceps"]
    pull_muscles = ["back", "biceps"]
    legs_muscles = ["legs", "glutes", "calves", "core"]
    rest = "60-90s" if profile.goal == "muscle_gain" else "30-60s"
    tempo = "2-1-2"
    main_count = "4 to 6"

    # Gemini often works well with markdown for lists and structure.
    # We are asking for JSON output, so the prompt structure guides it.
    prompt = f"""
    You are an expert AI personal trainer for MyFit Mantra.
    Your task is to create a structured, 12-session workout plan (4 weeks, 3 days/week) based on a user profile and a provided list of available exercises.
    Output MUST be a valid JSON object with a single key "plan", whose value is a LIST of 12 session objects.

    USER PROFILE:
    - Name: {profile.name}
    - Age: {profile.age}
    - Gender: {profile.gender}
    - Goal: {profile.goal}
    - Experience Level: {profile.experience}
    - Available Equipment: {profile.equipment}
    - Days Per Week: {profile.days_per_week}

    AVAILABLE EXERCISES (JSON LIST):
    ```json
    {json.dumps(exercise_list, indent=None)}
    ```

    DETAILED INSTRUCTIONS FOR THE PLAN:
    1.  **EXERCISES**: Use ONLY exercises present in the `AVAILABLE EXERCISES` list. Do not invent exercises or use equipment not available.
    2.  **STRUCTURE**: Generate exactly 12 sessions. Each session MUST have `warmup`, `main`, and `cooldown` sections.
    3.  **SECTIONS**:
        - `warmup`: Select 2-3 exercises where `type` is 'warmup'.
        - `main`: Select {main_count} exercises where `type` is 'main' or 'core'. Add `rest: "{rest}"` and `tempo: "{tempo}"` to each 'main' exercise.
        - `cooldown`: Select 2-3 exercises where `type` is 'cooldown'.
    4.  **WORKOUT SPLIT**: Implement a PUSH-PULL-LEGS rotation over the 12 sessions:
        - PUSH days (Sessions 1, 4, 7, 10): 'main' exercises must target: {", ".join(push_muscles)}.
        - PULL days (Sessions 2, 5, 8, 11): 'main' exercises must target: {", ".join(pull_muscles)}.
        - LEGS/CORE days (Sessions 3, 6, 9, 12): 'main' exercises must target: {", ".join(legs_muscles)}.
        - Ensure variety; do not use the exact same main exercises every PUSH day, etc., if alternatives exist.
     5. **PROGRESSIVE OVERLOAD**: Apply progressive overload week-to-week (e.g., compare Week 2 PUSH to Week 1 PUSH).
        - Increase reps slightly (e.g., 1-2 reps) OR
        - Increase sets slightly (e.g., add 1 set) OR
        - Increase duration slightly (e.g., + 5-10 sec) OR
        - Select a slightly more challenging variation if available in the list.
        Start Week 1 based on the sets/reps in the source data or reasonable defaults for the user's level/goal.
    6.  **SETS/REPS/DURATION**:
        - If the source exercise `reps` contains "sec" or "min" (e.g., "30 sec", "1 min"), put that value (or the overload value) into the output `duration` key, and omit `sets`/`reps` in the output object.
        - If the source exercise `reps` is a number (or a string representing a number that can be converted to an int), use the source `sets` and `reps` (adjusted for overload) in the output object and omit `duration`.
        - Use default sets/reps/duration from the exercise list, adjusting ONLY for progressive overload.
    7.  **OUTPUT FORMATTING**:
        - The entire output MUST be a single, valid JSON object.
        - The JSON object must have one top-level key: "plan".
        - The value of "plan" must be a LIST of exactly 12 session objects.
        - Each session object in the list MUST include: `session` (integer, 1-12), `focus` ("Push", "Pull", or "Legs"), and `sections` (an object with `warmup`, `main`, `cooldown` arrays).
        - Example structure for ONE session object within the list:
          {{
            "session": 1,
            "focus": "Push",
            "sections": {{
              "warmup": [ {{"name": "Ex Name", "duration": "2 min"}}, {{"name": "Ex Name", "sets": 2, "reps": 15}} ],
              "main": [ {{"name": "Ex Name", "sets": 3, "reps": 10, "rest": "{rest}", "tempo": "{tempo}" }} ],
              "cooldown": [ {{"name": "Ex Name", "duration": "1 min"}} ]
            }}
          }}
       - Do NOT include the `date` key; it will be added later by the Python backend.
       - Ensure the `session` number field is correctly populated sequentially from 1 to 12.
       - Ensure the `focus` field ("Push", "Pull", or "Legs") is correctly assigned.
       - Do not add any text before or after the JSON object. Just the JSON.

    Begin JSON output now:
    """
    return prompt

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # (This function remains the same as before)
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        LOG.error("static/index.html not found.")
        raise HTTPException(status_code=404, detail="Frontend HTML file not found.")
    except Exception as e:
        LOG.error(f"Error serving frontend: {e}")
        raise HTTPException(status_code=500, detail="Could not serve frontend.")

@app.post("/generate-plan", response_model=WorkoutPlanResponse, status_code=status.HTTP_200_OK)
async def generate_workout_plan(profile: UserProfile):
    if profile.days_per_week != 3:
         raise HTTPException(status_code=400, detail="Only 3 days_per_week generating 12 sessions is currently supported.")

    LOG.info(f"Received request for: {profile.name}, Goal: {profile.goal}, Equipment: {profile.equipment}")

    available_exercises = filter_exercises(profile)
    if not available_exercises:
         LOG.warning("No suitable exercises found after filtering.")
         # Consider raising HTTPException here if it's critical, or let Gemini try.
         # For this setup, we'll let Gemini attempt and it might return an empty plan or an explanation.

    prompt_for_gemini = construct_gemini_prompt(profile, available_exercises)

    try:
        LOG.info(f"Calling Google Gemini API with model {AI_MODEL_GEMINI}...")
        # LOG.debug(f"Gemini Prompt: {prompt_for_gemini}") # Uncomment for debugging

        # For Gemini, the response_mime_type helps ensure JSON output
        # However, the model needs to be capable and the prompt clear.
        # "gemini-1.5-flash" and "gemini-1.5-pro" are good with this.
        # The new API for 1.5 models also directly supports JSON mode.
        response = gemini_model.generate_content(
            prompt_for_gemini,
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1, # Default is 1
                # stop_sequences=['...'], # If needed
                # max_output_tokens=4096, # Set in model init or here
                temperature=0.2, # For consistency
                response_mime_type="application/json" # Request JSON output
            )
        )

        # Gemini can sometimes wrap its JSON in ```json ... ``` if not perfectly prompted for raw JSON.
        # Or it might have issues if safety filters block parts of the response.
        if not response.parts:
            LOG.error(f"Gemini API call returned no parts. Prompt feedback: {response.prompt_feedback}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise HTTPException(status_code=502, detail=f"AI model blocked the request/response. Reason: {response.prompt_feedback.block_reason_message}")
            raise HTTPException(status_code=502, detail="AI model returned an empty response.")

        content = response.text # .text should directly give the string content
        LOG.info("Google Gemini API call successful.")
        # LOG.debug(f"Gemini Raw Output: {content[:300]}...")


    except genai.types.BlockedPromptException as bpe: # Specific Gemini exception
        LOG.error(f"Gemini API call blocked. Prompt probably violated safety settings: {bpe}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Or 502 Bad Gateway
            detail=f"AI model blocked the prompt due to safety settings: {bpe}",
        )
    except genai.types.StopCandidateException as sce: # Specific Gemini exception
        LOG.error(f"Gemini API call stopped unexpectedly: {sce}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI model generation stopped unexpectedly: {sce}",
        )
    except Exception as e: # Catch other potential google.generativeai errors or general errors
        LOG.error(f"Google Gemini API error or other unexpected error: {e}")
        # Check if it's an API key issue (often a PermissionDenied error)
        if "API_KEY_INVALID" in str(e) or "PermissionDenied" in str(e):
            detail_msg = "AI service error: Invalid or missing Gemini API Key. Please check your .env configuration."
            status_c = status.HTTP_401_UNAUTHORIZED
        elif "quota" in str(e).lower(): # Gemini might have quota messages too
            detail_msg = f"AI service error: Gemini API quota exceeded. Please check your Google Cloud project quotas. ({e})"
            status_c = 429 # Too Many Requests
        else:
            detail_msg = f"AI service error (Gemini): {e}"
            status_c = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(
            status_code=status_c,
            detail=detail_msg,
        )

    try:
        # Gemini with response_mime_type="application/json" should return raw JSON string.
        # If it's wrapped in markdown ```json ... ```, we might need to strip it.
        # Let's try direct parsing first.
        if content.strip().startswith("```json"):
            content = content.strip()[7:-3].strip() # Remove ```json\n and \n```
        elif content.strip().startswith("```"): # More generic markdown code block
             content = content.strip()[3:-3].strip()


        plan_data = json.loads(content)
        if "plan" not in plan_data or not isinstance(plan_data["plan"], list):
             LOG.error(f"Gemini response JSON missing 'plan' list or not a list. Content: {content[:300]}")
             raise json.JSONDecodeError("JSON from AI does not contain 'plan' list key or it's not a list.", content, 0)

        final_plan = plan_data["plan"]

        if len(final_plan) != 12:
            LOG.warning(f"Gemini returned {len(final_plan)} sessions, but 12 were expected. Plan might be incomplete. Content: {content[:300]}")
            # Potentially truncate/pad or raise error

        dates = generate_dates(len(final_plan))

        for i, session_item in enumerate(final_plan):
             session_item['session'] = i + 1
             if i < len(dates):
                 session_item['date'] = dates[i]
             else:
                 session_item['date'] = "N/A" # Should not happen

             if 'focus' not in session_item or session_item['focus'] not in ["Push", "Pull", "Legs"]:
                 focus_map = {0: "Push", 1: "Pull", 2: "Legs"}
                 session_item['focus'] = focus_map.get(i % 3, "N/A")
                 LOG.warning(f"Session {i+1} 'focus' was missing/invalid from AI, defaulted to {session_item['focus']}.")

        note = ("Plan generated by AI (Google Gemini). Exercises selected and structured by the model "
                "based on user profile and available data. Always consult a professional before starting any new workout routine.")
        return WorkoutPlanResponse(plan=final_plan, ai_model_used=AI_MODEL_GEMINI, notes=note)

    except json.JSONDecodeError as e:
         LOG.error(f"Gemini failed to return valid JSON or JSON was not as expected: {e}\nContent snippet: {content[:500]}...")
         raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI model (Gemini) failed to return expected JSON. Error: {str(e)}. Raw start: {content[:150]}...",
         )
    except Exception as e:
        LOG.error(f"Error processing Gemini AI response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing AI plan from Gemini: {str(e)}")

# Optional: For running directly with uvicorn
# import uvicorn
# if __name__ == "__main__":
#      uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)