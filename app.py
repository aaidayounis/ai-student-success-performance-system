"""AI Students Success & Performance – Flask app (no JavaScript, no ML).

Part 1: shows the analysis from the notebook (charts + insights).
Part 2: a form where you enter marks; the server compares them with the
        dataset statistics from the notebook and gives a rule-based report.
"""
from math import erf, sqrt
from flask import Flask, render_template, request

app = Flask(__name__)

# ---- Dataset statistics (copied from df.describe() in the notebook) ----------
STATS = {
    "quiz":       {"label": "Quiz average",      "mean": 59.38, "std": 15.93, "unit": "/100"},
    "midterm":    {"label": "Midterm score",     "mean": 59.35, "std": 17.83, "unit": "/100"},
    "attendance": {"label": "Attendance",        "mean": 76.79, "std": 14.12, "unit": "%"},
    "study":      {"label": "Study hours / day", "mean": 3.55,  "std": 1.75,  "unit": "h"},
    "sleep":      {"label": "Sleep hours / day", "mean": 6.83,  "std": 1.21,  "unit": "h"},
    "gpa":        {"label": "Previous GPA",      "mean": 2.99,  "std": 0.50,  "unit": "/4"},
}
FINAL_MEAN, FINAL_STD = 59.50, 18.93

# Insights taken from the notebook's charts
INSIGHTS = {
    "grades": "C is the most common grade (~1,060 students), followed by D and F. Only a small group reaches A or A+ (~110 students).",
    "avg": "Quiz, midterm and final averages are almost the same (about 59–60 out of 100), so the typical student performs consistently across assessments.",
    "internet": "Students with internet access average about 60 on the final exam versus about 56 without it. Access helps, but it does not guarantee better results.",
    "heatmap": "Quiz average (0.56) and midterm score (0.52) have the strongest link with the final exam. Previous GPA (0.35) and study hours (0.24) matter moderately; attendance (0.15) and assignments completed (0.05) matter least.",
    "sleep": "Differences are small: below 5 hours averages about 57, while 5–9 hours is about 59–60 and above 9 hours about 62. Very short sleep is slightly worse.",
    "study": "A clear upward trend: 0–2 h averages about 53, 2–4 h about 58, 4–6 h about 63 and above 6 h about 68.",
    "attendance": "Higher attendance goes with higher scores: about 53 below 50% attendance, rising to about 62 at 85–100%.",
    "assignments": "Students who finish 76–100% of assignments score about 61 on average, versus about 52 for those completing 26–50%.",
}

# Rule-of-thumb grade bands (assumption, not taken from the dataset)
BANDS = [(90, "A+"), (80, "A"), (70, "B"), (55, "C"), (45, "D"), (0, "F")]


def percentile(value, mean, std):
    """Approximate percentile using the normal curve (mean/std from the dataset)."""
    return round(50 * (1 + erf((value - mean) / (std * sqrt(2)))))


def grade_for(score):
    for cut, g in BANDS:
        if score >= cut:
            return g
    return "F"


def parse(name, lo, hi, required=False):
    raw = (request.form.get(name) or "").strip()
    if raw == "":
        if required:
            raise ValueError(f"Please enter a value for {name.replace('_', ' ')}.")
        return None
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name.replace('_', ' ').title()} must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"{name.replace('_', ' ').title()} must be between {lo} and {hi}.")
    return v


def analyse(form_vals):
    v = form_vals
    # Weighted average of the strongest predictors found in the heatmap
    parts, weights = [], []
    for key, w in (("quiz", 0.4), ("midterm", 0.4)):
        parts.append(v[key] * w); weights.append(w)
    if v["gpa"] is not None:
        parts.append(v["gpa"] * 25 * 0.2); weights.append(0.2)
    est = sum(parts) / sum(weights)
    est = round(est, 1)

    factors = []
    for key, s in STATS.items():
        if v.get(key) is None:
            continue
        val = v[key]
        diff = val - s["mean"]
        top = max(val, s["mean"], 1) * 1.0
        scale = {"gpa": 4, "study": 10, "sleep": 10}.get(key, 100)
        factors.append({
            "label": s["label"], "value": val, "mean": s["mean"], "unit": s["unit"],
            "pct_you": min(100, val / scale * 100), "pct_avg": min(100, s["mean"] / scale * 100),
            "diff": round(diff, 2), "above": diff >= 0,
            "percentile": percentile(val, s["mean"], s["std"]),
        })

    comp = None
    if v["assign_done"] is not None and v["assign_total"]:
        comp = round(v["assign_done"] / v["assign_total"] * 100, 1)

    tips, good = [], []
    def add(cond, bad_msg, ok_msg):
        (tips if cond else good).append(bad_msg if cond else ok_msg)

    if v["study"] is not None:
        add(v["study"] < 3.5, "Study hours are below the dataset average. Students studying 4–6 h/day averaged ~63 on the final and 6+ h about ~68.",
            "Your study time is at or above average – the dataset shows study hours rising with final scores.")
    if v["attendance"] is not None:
        add(v["attendance"] < 70, "Attendance is under 70%. Students above 85% averaged about 62 versus about 53 under 50%.",
            "Attendance is healthy; higher attendance goes with higher exam scores.")
    if comp is not None:
        add(comp < 75, f"You completed {comp}% of assignments. Those finishing 76–100% averaged ~61 on the final.",
            f"You completed {comp}% of assignments – in the top completion band.")
    if v["sleep"] is not None:
        add(v["sleep"] < 5, "Sleep is below 5 hours; that group had the lowest average final score (~57).",
            "Sleep is in a fine range. Its effect in the data is small compared to quizzes and study time.")
    add(v["quiz"] < FINAL_MEAN, "Quiz average is below the dataset average. It is the strongest link with the final exam (0.56), so improving it pays off most.",
        "Quiz average is above the dataset average – the strongest indicator of a good final exam.")
    add(v["midterm"] < FINAL_MEAN, "Midterm score is below average; it is the second strongest link with the final exam (0.52).",
        "Midterm score is above average – a good sign for the final.")
    if v["internet"] == "No":
        tips.append("Without internet access, students averaged ~56 vs ~60 with it. Try campus labs or offline resources.")

    return {
        "estimate": est, "grade": grade_for(est),
        "percentile": percentile(est, FINAL_MEAN, FINAL_STD),
        "vs_avg": round(est - FINAL_MEAN, 1),
        "factors": factors, "completion": comp, "tips": tips, "good": good,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            vals = {
                "quiz": parse("quiz", 0, 100, True),
                "midterm": parse("midterm", 0, 100, True),
                "study": parse("study", 0, 24),
                "attendance": parse("attendance", 0, 100),
                "assign_done": parse("assign_done", 0, 1000),
                "assign_total": parse("assign_total", 1, 1000),
                "sleep": parse("sleep", 0, 24),
                "gpa": parse("gpa", 0, 4),
                "internet": request.form.get("internet") or None,
            }
            if vals["assign_done"] is not None and vals["assign_total"] is not None and vals["assign_done"] > vals["assign_total"]:
                raise ValueError("Assignments completed cannot be more than total assignments.")
            result = analyse(vals)
        except ValueError as e:
            error = str(e)
    return render_template("index.html", insights=INSIGHTS, result=result, error=error, form=form, stats=STATS)


if __name__ == "__main__":
    app.run(debug=True)
