# MyFit Mantra AI - Workout Plan Generator

## Objective
Design a mini AI engine using FastAPI and the OpenAI API to generate a 12-session progressive workout plan based on user input, fulfilling the MyFit Mantra AI Intern Assignment requirements and specifically addressing the feedback to include a genuine AI component.

## AI Integration (Addressing Feedback)
This project addresses the core feedback by moving beyond simple rule-based logic.
The AI component is handled via the **OpenAI API (`gpt-3.5-turbo` or `gpt-4`)**.
The AI is meaningfully used for:
1.  **Interpretation**: Understanding the user's `goal`, `experience`, and constraints.
2.  **Dynamic Selection**: Choosing suitable exercises from a pre-filtered list (filtered by Python logic for equipment/level).
3.  **Structuring**:
    *   Organising exercises into `warmup`, `main`, and `cooldown` sections.
    *    Implementing the required workout split (Push-Pull-Legs rotation) by selecting appropriate `muscle_group` exercises for the 'main' section of each specific day.
4.  **Exercise Science Application**:
     * Applying **Progressive Overload** by logically increasing sets, reps, or duration across the 4 weeks.
	 * Setting appropriate `rest` and `tempo` parameters.
5.  **Data Mapping**: Intelligently deciding whether an exercise's metadata fits into `sets`/`reps` or `duration` based on the input data format (e.g., "15" vs "30 sec").

Python/Pandas is used for loading, cleaning, and strictly pre-filtering the exercise list based on user `equipment` and `experience` level. This pre-filtered list is then passed to the LLM via a detailed prompt, allowing the AI to focus on selection, structuring, splitting and progression, while reducing the risk of the AI "hallucinating" unavailable equipment or exercises.

## Requirements Met
- [x] Web API (Python FastAPI).
- [x] Accepts user profile as JSON input.
- [x] Generates a 12-day progressive workout plan (3 sessions/week, 4 weeks).
- [x] Each workout contains 3 sections: Warm-Up, Main Exercises, Cool-Down.
- [x] Uses data from the provided CSV.
- [x] **AI Component**: Integration of third-party AI service (OpenAI API).
- [x] Bonus: Generate progressive overload (via AI Prompt).
- [x] Bonus: Add logic to alternate between push/pull/legs days (via AI Prompt).
- [x] Bonus: Export workout plan to JSON (API default response).
 - [x] Additional: Deployment instructions structure.

## Setup & Run Locally

1.  **Clone/Create project structure**:
    ```
     myfit_ai_app/
     ├── main.py                       
     ├── 50_exercises_dataset (1).csv  
     ├── requirements.txt              
     ├── .env                          
     └── README.md 
    ```
2.  **Ensure CSV**: Make sure `50_exercises_dataset (1).csv` is in the root directory.
3.  **Create Virtual Env (Recommended)**:
    ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
4.  **Install Dependencies**:
      ```bash
       pip install -r requirements.txt
      ```
5.  **API Key**: Create the `.env` file in the root and add your key:
      ```bash
       OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE
      ```
6.  **Run Server**:
     ```bash
      uvicorn main:app --reload
     ```
7.  **Access API Docs**: Open your browser to `http://127.0.0.1:8000/docs`
    You can test the `/generate-plan` endpoint directly from the docs page using the "Try it out" button.

## Accessing the Frontend
After running the server (`uvicorn main:app --reload`):
1.  Open your browser.
2.  Navigate to `http://127.0.0.1:8000/`.
This will load an HTML form where you can input the user profile details. Click "Generate My Plan" to submit the data to the AI backend. The generated 12-session workout plan will be displayed on the page.

## Endpoint
*   **URL:** `/generate-plan`
*   **Method:** `POST`
*   **Description:** Accepts a user profile and returns an AI-generated 12-session plan.

### Sample Request Body (POST to `/generate-plan`)
```json
 {
  "name": "Aarav",
  "age": 35,
  "gender": "male",
  "goal": "muscle_gain",
  "experience": "intermediate",
  "equipment": ["dumbbells", "bench", "resistance_band"],
  "days_per_week": 3
}