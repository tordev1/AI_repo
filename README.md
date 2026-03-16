# Student Success Copilot

A hybrid AI system that helps students manage their academic workload by providing personalised risk assessment, study scheduling, and actionable recommendations.

Built for **4COSC016C — Introduction to AI** coursework.

## AI Components

### 1. Search-Based Planner (`ai/search_planner.py`)
Generates a weekly study schedule using two search algorithms:
- **A\* Search** — uses f(n) = g(n) + h(n), considers both path cost and heuristic
- **Greedy Best-First Search** — uses f(n) = h(n) only, faster but not guaranteed optimal

Both algorithms are compared side-by-side with node exploration metrics.

**Heuristic:** combines deadline urgency and subject difficulty weighting.

### 2. Rule-Based Expert System (`ai/rule_engine.py`)
22 rules covering stress, workload, confidence, and deadline proximity:
- **Forward chaining** — fires rules based on student facts to derive risk level and recommendations
- **Backward chaining** — given a conclusion (e.g. "high risk"), traces back which facts caused it

### 3. Machine Learning Model (`ai/ml_model.py`)
- **Random Forest Classifier** trained on 1200 synthetic student records
- Predicts risk level (low / medium / high) from student features
- Evaluation: accuracy, precision, recall, F1 score, confusion matrix
- Features: gender, workload, study_hours, confidence, stress, days_to_deadline

### 4. Fuzzy Logic (Optional Enhancement) (`ai/fuzzy_logic.py`)
- Fuzzy membership functions for stress and confidence
- Produces a continuous risk score (0–10) instead of crisp categories

## Tech Stack
- Python 3.13
- Streamlit (web UI)
- scikit-learn (ML model)
- pandas / numpy (data handling)
- matplotlib / seaborn (visualisations)
- scikit-fuzzy (fuzzy logic)

## Project Structure
```
student-success-copilot/
├── app.py                       # Streamlit main app (UI + orchestration)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/
│   └── synthetic_students.csv   # Generated training dataset (1200 records)
├── ai/
│   ├── __init__.py
│   ├── data_generator.py        # Synthetic dataset generator
│   ├── ml_model.py              # Random Forest training, prediction, evaluation
│   ├── rule_engine.py           # Forward + backward chaining expert system
│   ├── search_planner.py        # A* and Greedy search-based schedule planner
│   └── fuzzy_logic.py           # Fuzzy membership functions
└── models/
    ├── risk_model.pkl            # Trained ML model
    └── gender_encoder.pkl        # Label encoder for gender feature
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd student-success-copilot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic dataset
```bash
python ai/data_generator.py
```
This creates `data/synthetic_students.csv` with 1200 student records.

### 4. Train the ML model
```bash
python ai/ml_model.py
```
This trains the Random Forest model, prints evaluation metrics, and saves the model to `models/risk_model.pkl`.

### 5. Run the application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

## How to Use

1. **Step 1** — Enter your details: gender, number of modules, study hours per week, confidence level, stress level, and days to nearest deadline.
2. **Step 2** — Add your subjects with name, hours needed, and difficulty rating.
3. **Results** — The system displays:
   - ML risk prediction with probability chart
   - Rule-based risk assessment with fired rules and backward chaining explanation
   - Fuzzy logic risk score
   - Weekly study schedule (A* vs Greedy comparison)
   - Combined summary with recommendations

## Testing Individual Components

```bash
# Test data generator
python ai/data_generator.py

# Test ML model (prints accuracy, precision, recall, F1, confusion matrix)
python ai/ml_model.py

# Test rule engine (forward + backward chaining demo)
python ai/rule_engine.py

# Test search planner (A* vs Greedy demo)
python ai/search_planner.py

# Test fuzzy logic
python ai/fuzzy_logic.py
```
