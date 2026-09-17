"""
Crime Safety Detection — Gradio Dashboard
Run with:
    python app.py

Place crime_safety_dataset.csv in the same folder as this file.
"""

import pandas as pd
import gradio as gr

from graph_chart import (
    load_data,
    age_distribution_chart,
    age_bracket_chart,
    age_by_time_chart,
    age_by_crime_boxplot,
    city_crime_heatmap,
    crime_type_chart,
)
from model import prepare_features, build_pipeline, train_and_evaluate, FEATURE_COLUMNS

# ---------------------------------------------------------------
# Load data + train model once at startup
# ---------------------------------------------------------------
df = load_data()
X, y = prepare_features(df)
pipeline = build_pipeline()
pipeline, accuracy = train_and_evaluate(X, y, pipeline)
feature_columns = X.columns

# ---------------------------------------------------------------
# Overview stats
# ---------------------------------------------------------------
total_cases = len(df)
avg_age = round(df["victim_age"].mean(), 1)
top_crime = df["crime_type"].value_counts().idxmax()
top_city = df["city"].value_counts().idxmax()
most_targeted_bracket = df["age_bracket"].value_counts().idxmax()

overview_md = f"""
### Crime Safety Detection — Overview

| Metric | Value |
|---|---|
| Total cases | {total_cases} |
| Average victim age | {avg_age} |
| Most common crime type | {top_crime} |
| City with most cases | {top_city} |
| Most targeted age bracket | {most_targeted_bracket} |
"""

insights_md = """
### Key Insights

- **Victims aged 60+ are the most frequently targeted age group** across nearly all crime
  types and time periods (405 of 1000 cases).
- During **evening and night hours**, the average victim age rises to ~52 years
  (mean 51.9, median 52, mode 52) — noticeably older than the overall average.
- **Burglary** peaks sharply in the **morning** among the 60+ age group (20 cases) —
  likely reflecting increased vulnerability when homes are left with only elderly
  residents present during typical work hours.
- **Domestic Violence** shows a consistently high rate among the 60+ group throughout
  the day, except at night, when younger victims (30 and below) become more common.
- A city-vs-crime-type crosstab shows crime types are **fairly evenly distributed
  across cities** — no single city stands out as disproportionately affected by any
  one crime type.
"""

model_note_md = f"""
### About the Prediction Model

A Random Forest Classifier was trained to predict **crime type** from city, age bracket,
time period, gender, race, and state.

**Accuracy: {accuracy:.1%}** — only slightly above the random-guess baseline of 10%
(since there are 10 crime types). A crosstab of city vs. crime type confirmed crime
types are nearly uniformly distributed across cities, meaning these features don't
carry a strong signal for predicting crime type in this dataset. The tool below is
included for transparency, not because it is highly reliable.
"""


def predict_crime_type(city, age_bracket, time_period, gender, race, state):
    row = pd.DataFrame([{
        "city": city,
        "age_bracket": age_bracket,
        "time_period": time_period,
        "victim_gender": gender,
        "victim_race": race,
        "state": state,
    }])
    row_encoded = pd.get_dummies(row, columns=FEATURE_COLUMNS)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    prediction = pipeline.predict(row_encoded)[0]
    probs = pipeline.predict_proba(row_encoded)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    classes = pipeline.named_steps["model"].classes_
    top3 = "\n".join(f"- {classes[i]}: {probs[i]:.1%}" for i in top3_idx)
    return f"**Predicted crime type: {prediction}**\n\nTop 3 likely outcomes:\n{top3}"


# ---------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------
with gr.Blocks(title="Crime Safety Detection Dashboard") as demo:
    gr.Markdown("# 🚨 Crime Safety Detection Dashboard")
    gr.Markdown(
        "Exploratory analysis of 1,000 reported crime cases, with a focus on "
        "victim age patterns across time, crime type, and location."
    )

    with gr.Tab("Overview"):
        gr.Markdown(overview_md)
        gr.Markdown(insights_md)

    with gr.Tab("Age Analysis"):
        gr.Markdown("### Victim Age Patterns")
        with gr.Row():
            gr.Plot(age_distribution_chart(df))
            gr.Plot(age_bracket_chart(df))
        gr.Plot(age_by_time_chart(df))
        gr.Plot(age_by_crime_boxplot(df))

    with gr.Tab("Crime & City Patterns"):
        gr.Markdown("### Crime Type and City Breakdown")
        gr.Plot(crime_type_chart(df))
        gr.Plot(city_crime_heatmap(df))

    with gr.Tab("Predict Crime Type"):
        gr.Markdown(model_note_md)
        with gr.Row():
            city_in = gr.Dropdown(sorted(df["city"].unique()), label="City")
            age_in = gr.Dropdown(sorted(df["age_bracket"].unique()), label="Age Bracket")
            time_in = gr.Dropdown(sorted(df["time_period"].unique()), label="Time Period")
        with gr.Row():
            gender_in = gr.Dropdown(sorted(df["victim_gender"].unique()), label="Gender")
            race_in = gr.Dropdown(sorted(df["victim_race"].unique()), label="Race")
            state_in = gr.Dropdown(sorted(df["state"].unique()), label="State")
        predict_btn = gr.Button("Predict Crime Type", variant="primary")
        result_out = gr.Markdown()

        predict_btn.click(
            fn=predict_crime_type,
            inputs=[city_in, age_in, time_in, gender_in, race_in, state_in],
            outputs=result_out,
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"))