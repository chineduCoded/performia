# Performia
Data-driven student performance insights

The system uses synthetically generated Nigerian university academic data modeled on real grading structures, attendance policies, and performance distributions. This allows safe experimentation while preserving real-world behavior.

## Problems

### Risk Prediction
> “Predict whether a student is at academic risk.”

- Type: Binary classification
- Target: `is_at_risk`
     - `1 = at risk / failed`
     - `0 = not at risk / passed`

### Score Prediction
> “Can we predict a student’s final score?”

- Type: Regression
- Target: `final_score`

### Improvement Detection
> “Is this student improving, declining, or stable?”
- Type: Trend classification
- Target: `performance_trend (improving / declining / stable)`

## Generate Synthetic Data
`uv run -m scripts.generate_data`