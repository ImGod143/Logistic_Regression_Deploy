"""
GenAI Student Impact Predictor
Single-file Flask app (backend + HTML/CSS/JS) for logistic.pkl.

Local run:   python app.py
Render run:  gunicorn app:app
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic.pkl")

app = Flask(__name__)

# Loaded once at startup (not on every request).
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_ORDER = list(model.feature_names_in_)

# ---------------------------------------------------------------------------
# Category encodings.
# The model only accepts numbers, so each dropdown label maps to the integer
# code used when the model was trained. These codes ASSUME alphabetical label
# encoding (sklearn LabelEncoder). If you encoded differently, edit the codes
# below, nothing else needs to change.
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Major_Category": {
        "Arts & Humanities": 0,
        "Business": 1,
        "Engineering": 2,
        "Health Sciences": 3,
        "Natural Sciences": 4,
        "Social Sciences": 5,
    },
    "Primary_Use_Case": {
        "Brainstorming": 0,
        "Coding help": 1,
        "Concept explanation": 2,
        "Research": 3,
        "Writing and editing": 4,
    },
    "Paid_Subscription": {"No": 0, "Yes": 1},
    "Institutional_Policy": {"Allowed": 0, "Banned": 1, "Restricted": 2},
}

# Form layout: (section title, [(feature, label, kind, options)])
# kind: "number" -> options = dict(min, max, step, value, hint)
#       "select" -> options = key into CATEGORIES
SECTIONS = [
    ("Student profile", [
        ("Student_ID", "Student ID", "number",
         dict(min=0, step=1, value=0, hint="Optional, has no effect on the result")),
        ("Major_Category", "Major", "select", "Major_Category"),
        ("Year_of_Study", "Year of study", "number",
         dict(min=1, max=8, step=1, value=2, hint="1 = first year")),
        ("Pre_Semester_GPA", "GPA before the semester", "number",
         dict(min=0, max=10, step=0.01, value=3.0, hint="On your dataset's scale")),
    ]),
    ("GenAI usage", [
        ("Weekly_GenAI_Hours", "GenAI hours per week", "number",
         dict(min=0, max=100, step=0.5, value=8, hint="Hours")),
        ("Primary_Use_Case", "Main use", "select", "Primary_Use_Case"),
        ("Prompt_Engineering_Skill", "Prompting skill", "number",
         dict(min=0, max=10, step=1, value=5, hint="Score, 0 to 10")),
        ("Tool_Diversity", "Number of AI tools used", "number",
         dict(min=0, max=20, step=1, value=2, hint="Count")),
        ("Paid_Subscription", "Paid subscription", "select", "Paid_Subscription"),
        ("Perceived_AI_Dependency", "Perceived dependency on AI", "number",
         dict(min=0, max=10, step=1, value=5, hint="Score, 0 to 10")),
        ("Institutional_Policy", "School policy on AI", "select", "Institutional_Policy"),
    ]),
    ("Study habits and wellbeing", [
        ("Traditional_Study_Hours", "Study hours per week without AI", "number",
         dict(min=0, max=100, step=0.5, value=12, hint="Hours")),
        ("Anxiety_Level_During_Exams", "Exam anxiety", "number",
         dict(min=0, max=10, step=1, value=5, hint="Score, 0 to 10")),
    ]),
    ("Semester outcomes", [
        ("Post_Semester_GPA", "GPA after the semester", "number",
         dict(min=0, max=10, step=0.01, value=3.0, hint="On your dataset's scale")),
        ("Skill_Retention_Score", "Skill retention score", "number",
         dict(min=0, max=100, step=0.1, value=60, hint="Score, 0 to 100")),
    ]),
]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenAI Student Impact Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;700&family=Nunito+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
{% raw %}
:root {
  --paper: #F2F5FB;
  --surface: #FFFFFF;
  --ink: #17203A;
  --muted: #5B6685;
  --line: #D9DFEE;
  --indigo: #2E3FD1;
  --indigo-deep: #1A2370;
  --indigo-soft: #E4E8FF;
  --lime: #C6F135;
  --low: #52B8A8;
  --medium: #8E9BFF;
  --high: #F26B5B;
  --error: #B3261E;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Nunito Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 16px;
  line-height: 1.55;
}
h1, h2, h3, .display { font-family: "Sora", "Nunito Sans", system-ui, sans-serif; }

.layout {
  display: grid;
  grid-template-columns: 380px 1fr;
  min-height: 100vh;
}

/* ---------- Result panel ---------- */
.result-panel {
  background: var(--indigo-deep);
  color: #fff;
  padding: 48px 40px;
  position: sticky;
  top: 0;
  height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 28px;
  overflow-y: auto;
}
.result-panel h1 {
  font-size: 1.7rem;
  line-height: 1.2;
  margin: 0;
  font-weight: 700;
}
.result-panel .lede { color: #B9C0F0; margin: 8px 0 0; font-size: 0.98rem; }

.verdict {
  background: rgba(255,255,255,0.07);
  border-radius: 20px;
  padding: 28px 24px;
  min-height: 150px;
}
.verdict .small { color: #B9C0F0; font-size: 0.92rem; margin: 0 0 6px; }
.verdict .label {
  font-family: "Sora", sans-serif;
  font-weight: 700;
  font-size: 3.2rem;
  line-height: 1.05;
  margin: 0;
  color: var(--lime);
}
.verdict .empty { color: #B9C0F0; margin: 0; }
.verdict .error { color: #FFB4AB; margin: 0; font-weight: 600; }

.bars { display: grid; gap: 16px; }
.bar-row { display: grid; gap: 6px; }
.bar-head { display: flex; justify-content: space-between; font-weight: 600; font-size: 0.95rem; }
.bar-track { height: 12px; border-radius: 12px; background: rgba(255,255,255,0.12); overflow: hidden; }
.bar-fill { height: 100%; width: 0; border-radius: 12px; transition: width .6s cubic-bezier(.2,.8,.2,1); }
.bar-fill.Low { background: var(--low); }
.bar-fill.Medium { background: var(--medium); }
.bar-fill.High { background: var(--high); }
.bar-row.top .bar-head { color: var(--lime); }

.foot-note { margin-top: auto; color: #8F98D6; font-size: 0.85rem; }

/* ---------- Form ---------- */
.form-area { padding: 48px clamp(24px, 5vw, 72px) 72px; max-width: 980px; }
.form-area > h2 { font-size: 1.5rem; margin: 0 0 6px; }
.form-area > p { color: var(--muted); margin: 0 0 32px; }

fieldset {
  border: 0;
  margin: 0 0 36px;
  padding: 0;
}
legend {
  font-family: "Sora", sans-serif;
  font-weight: 700;
  font-size: 1.12rem;
  padding: 0;
  margin-bottom: 16px;
  color: var(--indigo-deep);
}
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px 22px; }

.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-weight: 700; font-size: 0.95rem; }
.field .hint { color: var(--muted); font-size: 0.85rem; }

input, select {
  font: inherit;
  color: var(--ink);
  background: var(--surface);
  border: 1.5px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px;
  width: 100%;
  min-height: 48px;
  transition: border-color .15s, box-shadow .15s;
}
input:hover, select:hover { border-color: #B6BFDD; }
input:focus-visible, select:focus-visible, button:focus-visible {
  outline: 3px solid var(--indigo);
  outline-offset: 2px;
  border-color: var(--indigo);
}
input.invalid { border-color: var(--error); }

.actions { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.btn {
  font: inherit;
  font-weight: 700;
  border: 0;
  border-radius: 14px;
  padding: 15px 28px;
  min-height: 52px;
  cursor: pointer;
  transition: transform .12s, background .15s;
}
.btn.primary { background: var(--indigo); color: #fff; }
.btn.primary:hover { background: #2434B8; }
.btn.primary:active { transform: translateY(1px); }
.btn.primary[disabled] { opacity: .6; cursor: wait; }
.btn.ghost { background: var(--indigo-soft); color: var(--indigo-deep); }
.btn.ghost:hover { background: #D5DBFF; }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .result-panel { position: static; height: auto; padding: 32px 24px; }
  .foot-note { margin-top: 0; }
  .grid { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; scroll-behavior: auto !important; }
}
{% endraw %}
</style>
</head>
<body>
<div class="layout">

  <aside class="result-panel" aria-live="polite">
    <div>
      <h1>GenAI student impact predictor</h1>
      <p class="lede">Enter a student's details to see which class the model predicts: Low, Medium or High.</p>
    </div>

    <div class="verdict" id="verdict">
      <p class="empty">Fill in the form and select Predict to see a result here.</p>
    </div>

    <div class="bars" id="bars" hidden></div>

    <p class="foot-note">Logistic regression model, scikit-learn. Probabilities show how confident the model is in each class.</p>
  </aside>

  <main class="form-area">
    <h2>Student details</h2>
    <p>All fields use the same units as the data the model was trained on.</p>

    <div id="predict-form">
      {% for title, fields in sections %}
      <fieldset>
        <legend>{{ title }}</legend>
        <div class="grid">
          {% for name, label, kind, opts in fields %}
          <div class="field">
            <label for="{{ name }}">{{ label }}</label>
            {% if kind == "select" %}
            <select id="{{ name }}" name="{{ name }}">
              {% for text in categories[opts] %}
              <option value="{{ text }}">{{ text }}</option>
              {% endfor %}
            </select>
            {% else %}
            <input type="number" id="{{ name }}" name="{{ name }}"
                   min="{{ opts.min }}" {% if opts.max is defined %}max="{{ opts.max }}"{% endif %}
                   step="{{ opts.step }}" value="{{ opts.value }}" inputmode="decimal" required>
            <span class="hint">{{ opts.hint }}</span>
            {% endif %}
          </div>
          {% endfor %}
        </div>
      </fieldset>
      {% endfor %}

      <div class="actions">
        <button type="button" class="btn primary" id="predict-btn">Predict</button>
        <button type="button" class="btn ghost" id="reset-btn">Reset form</button>
      </div>
    </div>
  </main>
</div>

<script>
{% raw %}
const btn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const verdict = document.getElementById("verdict");
const bars = document.getElementById("bars");
const fields = Array.from(document.querySelectorAll("#predict-form input, #predict-form select"));
const defaults = fields.map(f => f.value);

function showError(msg) {
  verdict.innerHTML = "";
  const p = document.createElement("p");
  p.className = "error";
  p.textContent = msg;
  verdict.appendChild(p);
  bars.hidden = true;
}

function showResult(data) {
  verdict.innerHTML = "";
  const small = document.createElement("p");
  small.className = "small";
  small.textContent = "Predicted class";
  const label = document.createElement("p");
  label.className = "label";
  label.textContent = data.prediction;
  verdict.append(small, label);

  bars.innerHTML = "";
  data.probabilities.forEach((item, i) => {
    const pct = (item.probability * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar-row" + (item.label === data.prediction ? " top" : "");
    row.innerHTML =
      '<div class="bar-head"><span></span><span></span></div>' +
      '<div class="bar-track"><div class="bar-fill"></div></div>';
    row.querySelectorAll(".bar-head span")[0].textContent = item.label;
    row.querySelectorAll(".bar-head span")[1].textContent = pct + "%";
    const fill = row.querySelector(".bar-fill");
    fill.classList.add(item.label);
    bars.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = pct + "%"; }));
  });
  bars.hidden = false;
}

btn.addEventListener("click", async () => {
  let valid = true;
  const payload = {};
  fields.forEach(f => {
    f.classList.remove("invalid");
    if (f.value === "" || (f.type === "number" && Number.isNaN(Number(f.value)))) {
      f.classList.add("invalid");
      valid = false;
    }
    payload[f.name] = f.value;
  });
  if (!valid) {
    showError("Some fields are empty. Fill in the highlighted fields and try again.");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Predicting...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) { showError(data.error || "The prediction failed. Try again."); }
    else { showResult(data); }
  } catch (err) {
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict";
  }
});

resetBtn.addEventListener("click", () => {
  fields.forEach((f, i) => { f.value = defaults[i]; f.classList.remove("invalid"); });
  verdict.innerHTML = '<p class="empty">Fill in the form and select Predict to see a result here.</p>';
  bars.hidden = true;
});
{% endraw %}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, sections=SECTIONS, categories=CATEGORIES)


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send the form values as JSON."), 400

    row = {}
    for name in FEATURE_ORDER:
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            return jsonify(error=f"Missing value for {name.replace('_', ' ')}."), 400
        if name in CATEGORIES:
            if raw not in CATEGORIES[name]:
                return jsonify(error=f"Unknown option for {name.replace('_', ' ')}."), 400
            row[name] = CATEGORIES[name][raw]
        else:
            try:
                row[name] = float(raw)
            except (TypeError, ValueError):
                return jsonify(error=f"{name.replace('_', ' ')} must be a number."), 400

    frame = pd.DataFrame([row], columns=FEATURE_ORDER)
    probabilities = model.predict_proba(frame)[0]
    prediction = str(model.predict(frame)[0])

    ordered = ["Low", "Medium", "High"]
    by_label = {str(c): float(p) for c, p in zip(model.classes_, probabilities)}
    return jsonify(
        prediction=prediction,
        probabilities=[
            {"label": c, "probability": by_label[c]} for c in ordered if c in by_label
        ],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
