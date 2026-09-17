"""
Crime Safety Detection — Chart & Visualization Functions
Generates matplotlib/seaborn figures used by both the notebook and the Gradio app.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "crime_safety_dataset.csv"


def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour

    def time_period(hour):
        if hour < 6:
            return "Night"
        elif hour < 12:
            return "Morning"
        elif hour < 18:
            return "Afternoon"
        else:
            return "Evening"

    df["time_period"] = df["hour"].apply(time_period)

    def age_bracket(age):
        if age <= 30:
            return "30 or Below"
        elif age <= 40:
            return "31-40"
        elif age <= 50:
            return "41-50"
        elif age <= 60:
            return "51-60"
        else:
            return "60+"

    df["age_bracket"] = df["victim_age"].apply(age_bracket)
    return df


def age_distribution_chart(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["victim_age"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Victim Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def age_bracket_chart(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = df["age_bracket"].value_counts().index
    sns.countplot(data=df, x="age_bracket", order=order, ax=ax)
    ax.set_title("Victims by Age Bracket")
    ax.set_xlabel("Age Bracket")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def age_by_time_chart(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Morning", "Afternoon", "Evening", "Night"]
    sns.countplot(data=df, x="time_period", hue="age_bracket", order=order, ax=ax)
    ax.set_title("Age Bracket Distribution Across Time Periods")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Count")
    ax.legend(title="Age Bracket", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def age_by_crime_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="crime_type", y="victim_age", ax=ax)
    ax.set_title("Age Distribution by Crime Type")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    return fig


def city_crime_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    city_crime = pd.crosstab(df["city"], df["crime_type"])
    sns.heatmap(city_crime, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_title("Crime Type by City")
    fig.tight_layout()
    return fig


def crime_type_chart(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df["crime_type"].value_counts().index
    sns.countplot(data=df, x="crime_type", order=order, ax=ax)
    ax.set_title("Crimes by Type")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_data()
    age_distribution_chart(data)
    age_bracket_chart(data)
    age_by_time_chart(data)
    age_by_crime_boxplot(data)
    city_crime_heatmap(data)
    crime_type_chart(data)
    plt.show()
