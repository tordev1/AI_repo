# Student Success Copilot — Summary Report

**Module:** 4COSC016C — Introduction to AI
**Team Members:** 00020472, 00021430
**Project:** Hybrid AI System — Student Success Copilot

---

## 1. Introduction

The goal of this project was to build a hybrid AI system that helps university students manage their academic workload. We named it the Student Success Copilot. The system takes basic student information such as stress level, confidence, workload, study hours, gender, and days to the nearest deadline, and produces three outputs: a personalised weekly study plan, a risk level prediction, and a human-readable explanation of its advice.

We chose this project because it directly relates to our own experience as students. We often struggle with balancing multiple deadlines and managing stress, so building a tool that addresses this felt both practical and meaningful. The system combines multiple AI techniques into a single hybrid architecture, demonstrating that no single AI approach can solve a complex real-world problem on its own.

## 2. Our Approach and Journey

We began by studying the coursework brief and identifying the three required AI components: a search-based planner, a rule-based expert system, and a machine learning model. We also decided to implement fuzzy logic as an optional enhancement to strengthen our submission.

Our first step was research. We reviewed the lecture materials on search algorithms, particularly A* and Greedy Best-First Search, to understand how heuristic functions guide the search process. We then studied forward and backward chaining from the knowledge-based systems lectures, paying close attention to how Modus Ponens is applied iteratively to derive new facts. For the machine learning component, we revisited the lessons on data preparation, train-test splitting, and evaluation metrics such as accuracy, precision, recall, and F1 score.

We used AI-assisted coding tools throughout the development process. These tools helped us generate initial code structures, debug errors, and explore alternative implementations. However, we made sure to understand every component before integrating it into the final system. Every function, every rule, and every algorithm was reviewed and tested by us to ensure correctness and alignment with what we learned in lectures.

## 3. AI Components

### 3.1 Search-Based Study Planner

The study planner models scheduling as a state-space search problem. The state is a partially filled weekly schedule, the actions are allocating one hour of a subject to a day, and the goal state is reached when all subjects have their required hours allocated. We implemented two search algorithms and compared them.

A* Search uses the evaluation function f(n) = g(n) + h(n), where g(n) is the cost accumulated so far and h(n) is a heuristic estimate of the remaining cost. Our heuristic combines deadline urgency and subject difficulty, giving higher priority to subjects with closer deadlines and higher difficulty ratings. Because A* considers both the path cost and the heuristic, it produces optimal schedules.

Greedy Best-First Search uses only the heuristic: f(n) = h(n). It is faster because it ignores the cost of the path taken so far, but it does not guarantee an optimal solution. In our testing, both algorithms produced identical schedules for small inputs, but A* would outperform Greedy on more complex scheduling scenarios where path cost matters.

### 3.2 Rule-Based Expert System

We built a rule-based expert system with 22 IF-THEN rules organised in two layers. The first layer classifies raw student inputs into categories. For example, if stress is 8 or above, the system derives stress_level as very_high. If confidence is 3 or below, it derives confidence_level as very_low.

The second layer combines these derived facts to determine risk levels and generate recommendations. For instance, Rule R17 states: if stress_level is very_high or high AND confidence_level is very_low or low, then risk_flag is high, and the system recommends seeking academic support.

Forward chaining starts with the student's input facts and fires rules iteratively until no new facts can be derived. This is a data-driven approach. Backward chaining works in the opposite direction: given a conclusion such as risk is high, it traces back through the rules to identify which specific input facts caused that conclusion. This provides explainability, allowing the student to understand why the system made a particular assessment.

### 3.3 Machine Learning Model

We generated a synthetic dataset of 1,200 student records using realistic statistical distributions. Each record contains six features: gender, workload, study hours, confidence, stress, and days to deadline. The risk label was derived using a scoring function that mirrors real-world patterns, with added noise to prevent perfect separability.

We trained a Random Forest Classifier with 100 decision trees. Random Forest was chosen because it works well with small datasets, handles both numerical and categorical features, provides interpretable feature importance, and is resistant to overfitting. We used an 80/20 stratified train-test split.

The model achieved 76.7% accuracy, 77.2% precision, 76.7% recall, and 76.8% F1 score. Importantly, the confusion matrix showed that the model never misclassifies high-risk students as low-risk, which is critical for a safety-oriented application. Feature importance analysis revealed that workload and stress are the strongest predictors at 24.2% each, while gender has minimal impact at 2.8%.

### 3.4 Fuzzy Logic (Optional Enhancement)

Traditional rules use crisp boundaries, meaning a stress level of 7.9 is categorised as high while 8.0 becomes very_high. Fuzzy logic addresses this limitation by allowing partial membership in multiple categories simultaneously. We defined triangular membership functions for stress and confidence, and created 16 fuzzy rules that map combinations to a continuous risk score from 0 to 10. This provides a more nuanced assessment than crisp categories alone.

## 4. System Integration

All four AI components are integrated through a Streamlit web interface. The user fills in their details in Step 1, adds their subjects in Step 2, and receives a comprehensive results page that displays the ML prediction with probability charts, rule-based risk assessment with fired rules, backward chaining explanation, fuzzy logic risk score, and two study schedules generated by A* and Greedy search shown side by side with a comparison table. A combined summary merges all outputs into actionable advice.

## 5. Reflection and Evaluation

This project taught us that hybrid AI systems are more powerful than any single technique. The ML model is good at prediction but cannot explain its reasoning. The rule engine provides clear explanations but relies on manually defined rules. The search planner generates actionable plans but needs the other components to determine what to prioritise. Together, they complement each other.

We used AI tools responsibly as permitted by the module policy. We used them to generate and debug code, but we ensured we understood every component, tested edge cases, and verified that the system behaves correctly. The code, rules, and dataset were all reviewed and validated by our team.

In total, we implemented six AI techniques across four categories: A* Search, Greedy Best-First Search, Forward Chaining, Backward Chaining, Random Forest Classification, and Fuzzy Logic Inference. This exceeds the minimum requirement of three AI components, and we believe the system demonstrates a strong understanding of how different AI approaches can work together in a practical application.

**Word count: ~1,000**
