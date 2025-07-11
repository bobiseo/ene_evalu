from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
import pandas as pd
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    df = get_input_data()
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)

    # ── NEW: translate summaries & content into English ──
    df[Config.INTERACTION_CONTENT] = translate_to_en(df[Config.INTERACTION_CONTENT].tolist())
    df[Config.TICKET_SUMMARY]       = translate_to_en(df[Config.TICKET_SUMMARY].tolist())


    # 🔥 labels
    df["intent"] = df["y2"]
    df["tone"] = df["y3"]
    df["resolution"] = df["y4"]

    return df

def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def compute_custom_accuracy(df):
    scores = []
    for _, row in df.iterrows():
        intent_correct = row["intent"] == row["predicted_intent"]
        tone_correct = row["tone"] == row["predicted_tone"]
        resolution_correct = row["resolution"] == row["predicted_resolution"]

        if not intent_correct:
            scores.append(0)
        elif not tone_correct:
            scores.append(1/3)
        elif not resolution_correct:
            scores.append(2/3)
        else:
            scores.append(1)

    df["score"] = scores
    accuracy = sum(scores) / len(scores)
    return df, accuracy

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)

    # save cleaned data
    df.to_csv("cleaned_data_preview.csv", index=False)
    print("✅ Saved data: cleaned_data_preview.csv")

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        print(f"\n[Group: {name}]")

        # ----- [step1] Intent prediction -----
        group_df["y"] = group_df["intent"]
        X_intent, group_df = get_embeddings(group_df)
        data_intent = get_data_object(X_intent, group_df)
        intent_preds = model_predict(data_intent, group_df, name="Intent")
        print("✅ Intent Predictions Sample:", intent_preds[:5])

        predicted_series = pd.Series([None] * len(group_df), index=group_df.index)
        predicted_series.iloc[:len(intent_preds)] = intent_preds
        group_df["predicted_intent"] = predicted_series

        # ----- [step2] Tone prediction -----
        group_df["y"] = group_df["tone"]
        X_tone, group_df = get_embeddings(group_df)
        data_tone = get_data_object(X_tone, group_df)
        tone_preds = model_predict(data_tone, group_df, name="Tone")
        print("✅ Tone Predictions Sample:", tone_preds[:5])

        predicted_series = pd.Series([None] * len(group_df), index=group_df.index)
        predicted_series.iloc[:len(tone_preds)] = tone_preds
        group_df["predicted_tone"] = predicted_series

        # ----- [step3] Resolution prediction -----
        group_df["y"] = group_df["resolution"]
        X_resolution, group_df = get_embeddings(group_df)
        data_resolution = get_data_object(X_resolution, group_df)
        resolution_preds = model_predict(data_resolution, group_df, name="Resolution")
        print("✅ Resolution Predictions Sample:", resolution_preds[:5])

        predicted_series = pd.Series([None] * len(group_df), index=group_df.index)
        predicted_series.iloc[:len(resolution_preds)] = resolution_preds
        group_df["predicted_resolution"] = predicted_series

        # save results
        output_file = f"predictions_{name}.csv".replace(" ", "_").replace("&", "and")
        group_df.to_csv(output_file, index=False)
        print(f"✅ saved prediction CSV : {output_file}")

        # compute and display conditional accuracy
        group_df, accuracy = compute_custom_accuracy(group_df)
        print(f"🎯 conditional accuracy (Group: {name}): {accuracy:.2%}")

        output_file_scored = output_file.replace(".csv", "_scored.csv")
        group_df.to_csv(output_file_scored, index=False)
        print(f"✅ saved prediction with score CSV : {output_file_scored}")
