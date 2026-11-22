import streamlit as st
import csv
import math

st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

FEATURES = ["Attendance", "Study_Hours", "Assignment_Score", "Internal_Marks", "Previous_Semester_Marks"]

def solve_linear_system(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[pivot] = M[pivot], M[i]
        for r in range(i + 1, n):
            factor = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= factor * M[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x

@st.cache_data
def load_and_train():
    seen = set()
    rows = []
    with open("dataset/student_performance_1000.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = tuple(r.items())
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    cols = FEATURES + ["Final_Marks"]
    medians = {}
    for col in cols:
        vals = sorted(float(r[col]) for r in rows if r.get(col) not in ("", None, "NA", "NaN", "nan"))
        mid = len(vals) // 2
        medians[col] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    X, y = [], []
    for r in rows:
        x_row = [1.0]
        for col in FEATURES:
            val = r.get(col)
            x_row.append(medians[col] if val in ("", None, "NA", "NaN", "nan") else float(val))
        y_val = r.get("Final_Marks")
        y.append(medians["Final_Marks"] if y_val in ("", None, "NA", "NaN", "nan") else float(y_val))
        X.append(x_row)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    k = len(X_train[0])
    XtX = [[sum(X_train[i][r] * X_train[i][c] for i in range(len(X_train))) for c in range(k)] for r in range(k)]
    Xty = [sum(X_train[i][r] * y_train[i] for i in range(len(X_train))) for r in range(k)]
    weights = solve_linear_system(XtX, Xty)

    preds = [sum(weights[j] * row[j] for j in range(k)) for row in X_test]
    mean_y = sum(y_test) / len(y_test)
    ss_res = sum((y_test[i] - preds[i]) ** 2 for i in range(len(y_test)))
    ss_tot = sum((y_test[i] - mean_y) ** 2 for i in range(len(y_test)))

    metrics = {
        "r2": 1 - (ss_res / ss_tot),
        "mae": sum(abs(y_test[i] - preds[i]) for i in range(len(y_test))) / len(y_test),
        "rmse": math.sqrt(ss_res / len(y_test))
    }
    return weights, metrics

weights, metrics = load_and_train()

st.title("🎓 Student Academic Performance Analysis & Prediction")
st.markdown("Interactive Multiple Linear Regression Predictor & Analytical Dashboard (**1,000 Student Records**)")

with st.expander("👥 Project Team Members", expanded=False):
    st.table([
        {"Member": "Member 1", "Name": "Ahmed Shariff.A", "Register Number": "311824102302"},
        {"Member": "Member 2", "Name": "Mohamed Sahith", "Register Number": "311824102006"},
        {"Member": "Member 3", "Name": "Mohammed Riyaz", "Register Number": "311824102045"},
        {"Member": "Member 4", "Name": "Ahamed Shariff.K", "Register Number": "311824102006"},
    ])

st.divider()

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🎛️ Enter Student Academic & Behavioral Inputs")
    attendance = st.slider("Attendance (%)", min_value=40.0, max_value=100.0, value=85.0, step=0.5)
    study_hours = st.slider("Daily Study Hours", min_value=0.5, max_value=10.0, value=4.5, step=0.5)
    assignment = st.slider("Assignment Score (out of 100)", min_value=30.0, max_value=100.0, value=82.0, step=1.0)
    internal = st.slider("Internal Marks (out of 100)", min_value=30.0, max_value=100.0, value=80.0, step=1.0)
    prev_sem = st.slider("Previous Semester Marks (out of 100)", min_value=30.0, max_value=100.0, value=78.0, step=1.0)

with col2:
    st.subheader("🎯 Predicted Final Examination Marks")
    x_in = [1.0, attendance, study_hours, assignment, internal, prev_sem]
    raw_pred = sum(weights[j] * x_in[j] for j in range(len(weights)))
    predicted_marks = max(0.0, min(100.0, raw_pred))

    st.metric(label="Predicted Final Marks", value=f"{predicted_marks:.2f} / 100")

    if predicted_marks >= 80:
        st.success("Grade Standing: Distinction / High Performance")
    elif predicted_marks >= 60:
        st.info("Grade Standing: First Class / Good Standing")
    elif predicted_marks >= 50:
        st.warning("Grade Standing: Pass / Needs Moderate Improvement")
    else:
        st.error("Grade Standing: At Risk — Academic Intervention Recommended")

    st.markdown("### 📊 Live Feature Impact on Score")
    impact_data = {
        "Attendance": round(weights[1] * attendance, 2),
        "Study Hours": round(weights[2] * study_hours, 2),
        "Assignments": round(weights[3] * assignment, 2),
        "Internals": round(weights[4] * internal, 2),
        "Prev Semester": round(weights[5] * prev_sem, 2),
    }
    st.bar_chart(impact_data, height=220)

    st.markdown("### 🧪 Model Evaluation Metrics (Test Set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{metrics['r2']:.4f}")
    m2.metric("MAE", f"{metrics['mae']:.2f}")
    m3.metric("RMSE", f"{metrics['rmse']:.2f}")

st.divider()

st.subheader("📈 Statistical & Mathematical Analysis Visualizations")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Marks Distribution",
    "2. Law of Large Numbers",
    "3. Correlation Matrix",
    "4. Gradient Descent",
    "5. Residuals Analysis"
])

with tab1:
    st.image("plots/marks_distribution.png", use_container_width=True)
with tab2:
    st.image("plots/law_of_large_numbers.png", use_container_width=True)
with tab3:
    st.image("plots/correlation_heatmap.png", use_container_width=True)
with tab4:
    st.image("plots/gradient_descent_cost.png", use_container_width=True)
with tab5:
    st.image("plots/actual_vs_predicted_residuals.png", use_container_width=True)