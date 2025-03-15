import functools
import os
import re
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from youtube_transcript_api import (CouldNotRetrieveTranscript, InvalidVideoId,
                                    NoTranscriptFound, TranscriptsDisabled,
                                    VideoUnavailable, YouTubeTranscriptApi)

# Environment setup
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Constants and Configuration
class Constants(Enum):
    MAX_TOKEN_LENGTH = 512
    DEFAULT_CHUNK_SIZE = 128
    DEFAULT_TOP_N_KEYWORDS_CHUNK = 5
    DEFAULT_TOP_N_KEYWORDS_OVERALL = 10
    BIAS_COLORS = ["blue", "gray", "red"]
    HATE_COLORS = ["red", "green"]
    SENTIMENT_COLORS = ["#FF0000", "#FF6666", "#808080", "#66FF66", "#00FF00"]


@dataclass(frozen=True)
class AnalysisConfig:
    stop_words: set = field(default_factory=lambda: set(stopwords.words("english")))
    stemmer: PorterStemmer = field(default_factory=PorterStemmer)
    lemmatizer: WordNetLemmatizer = field(default_factory=WordNetLemmatizer)
    url_token: str = "<URL>"
    mention_token: str = "<MENTION>"
    youtube_regex: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"""(?:https?://)?(?:www\.)?(?:youtube|youtu|youtube-nocookie)[.](?:com|be)/
            (?:watch\?v=|embed/|v/|.+\?v=)?([^&=%?]{11})""",
            re.VERBOSE,
        )
    )
    abbreviations: frozenset = field(
        default_factory=lambda: frozenset(
            {
                "Mr.",
                "Ms.",
                "Mrs.",
                "Dr.",
                "Prof.",
                "St.",
                "Ave.",
                "etc.",
                "U.S.A.",
                "U.K.",
            }
        )
    )


config = AnalysisConfig()


# Model Definitions
MODELS = {
    "politicalBiasBERT": {
        "model_name": "bucketresearch/politicalBiasBERT",
        "labels": ["left", "center", "right"],
        "task": "bias",
        "description": "Detects political bias (left, center, right).",
    },
    "roberta-hate-speech": {
        "model_name": "facebook/roberta-hate-speech-dynabench-r4-target",
        "labels": ["nothate", "hate"],
        "task": "hate_speech",
        "description": "Detects hate speech.",
    },
    "bert-base-multilingual-uncased-sentiment": {
        "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "labels": ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
        "task": "sentiment",
        "description": "Detects sentiment.",
    },
    "english-sarcasm-detector": {
        "model_name": "helinivan/english-sarcasm-detector",
        "labels": ["NOT_SARCASM", "SARCASM"],
        "task": "sarcasm",
        "description": "Detects sarcasm in text (BERT-based).",
    },
}


# Initialization Functions
def ensure_nltk_resources():
    """Ensure NLTK resources are downloaded only once."""
    resources = ["stopwords", "averaged_perceptron_tagger", "punkt", "wordnet"]
    for resource in resources:
        with suppress(LookupError):
            nltk.data.find(f"{resource.split('/')[0]}/{resource.split('/')[-1]}")
            continue
        nltk.download(resource.split("/")[-1])


@st.cache_resource
def load_model(model_name: str) -> Tuple:
    """Load and cache a model and tokenizer."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), tokenizer, device


# Helper Functions
def classify_text(
    text: str, model, tokenizer, device: str, labels: List[str]
) -> Tuple[str, float]:
    """Classify text using a given model and return label and confidence."""
    import torch
    from torch.nn.functional import softmax

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=Constants.MAX_TOKEN_LENGTH.value,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    idx = probs.argmax().item()
    return labels[idx], float(probs[0, idx])


def clean_text(
    text: str, do_stemming: bool = False, do_lemmatization: bool = True
) -> str:
    """Clean text by removing URLs, mentions, punctuation, and optionally stemming or lemmatizing."""
    text = re.sub(r"https?://\S+|www\.\S+", config.url_token, text)
    text = re.sub(r"@\w+", config.mention_token, text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text).lower()

    words = word_tokenize(text)
    if do_lemmatization:
        text = " ".join(config.lemmatizer.lemmatize(word) for word in words)
    elif do_stemming:
        text = " ".join(config.stemmer.stem(word) for word in words)
    return text


def extract_keywords_tfidf(text_chunks: List[str], top_n: int = 5) -> List[List[str]]:
    """Extract top keywords from text chunks using TF-IDF."""
    filtered_chunks = [
        " ".join(
            word for word in clean_text(chunk).split() if word not in config.stop_words
        )
        for chunk in text_chunks
        if any(word not in config.stop_words for word in clean_text(chunk).split())
    ]
    if not filtered_chunks:
        return [[] for _ in text_chunks]

    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(filtered_chunks)
    feature_names = np.array(vectorizer.get_feature_names_out())
    return [
        feature_names[tfidf_matrix[i].toarray().argsort()[0][-top_n:][::-1]].tolist()
        for i in range(tfidf_matrix.shape[0])
    ]


def analyze_pronoun_usage(text: str) -> Dict[str, int]:
    """Analyze pronoun usage in text."""
    pronoun_counts = {
        "first_person_singular": 0,
        "first_person_plural": 0,
        "second_person": 0,
        "third_person_singular": 0,
        "third_person_plural": 0,
    }
    pronoun_map = {
        "i": "first_person_singular",
        "me": "first_person_singular",
        "my": "first_person_singular",
        "mine": "first_person_singular",
        "we": "first_person_plural",
        "us": "first_person_plural",
        "our": "first_person_plural",
        "ours": "first_person_plural",
        "you": "second_person",
        "your": "second_person",
        "yours": "second_person",
        "he": "third_person_singular",
        "him": "third_person_singular",
        "his": "third_person_singular",
        "she": "third_person_singular",
        "her": "third_person_singular",
        "hers": "third_person_singular",
        "it": "third_person_singular",
        "its": "third_person_singular",
        "they": "third_person_plural",
        "them": "third_person_plural",
        "their": "third_person_plural",
        "theirs": "third_person_plural",
    }
    tagged_words = nltk.pos_tag(word_tokenize(text))
    for word, _ in tagged_words:
        word = word.lower()
        if word in pronoun_map:
            pronoun_counts[pronoun_map[word]] += 1
    return pronoun_counts


def analyze_speech_chunk(
    text_chunk: str,
    selected_model_name: str,
    model,
    tokenizer,
    device: str,
) -> Dict:
    """Analyze a single chunk of text."""
    output, confidence = classify_text(
        text_chunk, model, tokenizer, device, MODELS[selected_model_name]["labels"]
    )
    cleaned_chunk = clean_text(text_chunk)
    keywords = extract_keywords_tfidf([cleaned_chunk])[0]
    pronoun_counts = analyze_pronoun_usage(text_chunk)
    sentiment = (
        classify_text(
            text_chunk,
            model,
            tokenizer,
            device,
            MODELS["bert-base-multilingual-uncased-sentiment"]["labels"],
        )[0]
        if MODELS[selected_model_name]["task"] == "sentiment"
        else "N/A"
    )
    return {
        "text": text_chunk,
        "sentiment": sentiment,
        "confidence": confidence,
        "keywords": keywords,
        "pronoun_counts": pronoun_counts,
        "output": output,
        "output_confidence": confidence,
    }


def chunk_speech(
    speech_text: str, chunk_size: int = Constants.DEFAULT_CHUNK_SIZE.value
) -> List[str]:
    """Chunk speech text into smaller segments."""
    return [" ".join(sent.split()[:chunk_size]) for sent in sent_tokenize(speech_text)]


@st.cache_data
def analyze_speech(
    speech_text: str,
    selected_model_name: str,
    chunk_size: int = Constants.DEFAULT_CHUNK_SIZE.value,
    top_n_keywords_chunk: int = Constants.DEFAULT_TOP_N_KEYWORDS_CHUNK.value,
    top_n_keywords_overall: int = Constants.DEFAULT_TOP_N_KEYWORDS_OVERALL.value,
) -> pd.DataFrame:
    """Analyze the full speech text."""
    model, tokenizer, device = load_model(MODELS[selected_model_name]["model_name"])
    chunks = chunk_speech(speech_text, chunk_size)
    analyzed_chunks = (
        analyze_speech_chunk(chunk, selected_model_name, model, tokenizer, device)
        for chunk in chunks
        if chunk.strip()
    )
    df = pd.DataFrame(tqdm(analyzed_chunks, total=len(chunks), desc="Analyzing Chunks"))
    if not df.empty:
        overall_keywords = list(
            chain.from_iterable(extract_keywords_tfidf(chunks, top_n_keywords_overall))
        )
        df["overall_keywords"] = [overall_keywords] * len(df)
    else:
        df["overall_keywords"] = [[]]
    return df


# YouTube Transcript Fetching
@functools.lru_cache(maxsize=128)
def fetch_transcript(video_id: str, languages: Tuple[str, ...] = ("en",)) -> List[Dict]:
    """Fetch raw transcript data with caching."""
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_transcript(languages)
    return transcript.fetch()


@st.cache_data
def get_youtube_transcript(url: str, languages: Tuple[str, ...] = ("en",)) -> str:
    """Fetch and preprocess YouTube transcript, caching by URL."""
    video_id = extract_video_id(url)
    if not video_id:
        return ""
    try:
        fetched_transcript = fetch_transcript(video_id, languages)
        transcript_text = " ".join(entry["text"] for entry in fetched_transcript)
        return improve_transcript_punctuation(transcript_text)
    except (
        NoTranscriptFound,
        TranscriptsDisabled,
        CouldNotRetrieveTranscript,
        InvalidVideoId,
        VideoUnavailable,
    ) as e:
        st.error(f"Error fetching transcript: {type(e).__name__} - {e}")
        return ""


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    match = config.youtube_regex.match(url)
    if match:
        return match.group(1)
    parsed_url = urlparse(url)
    if parsed_url.netloc == "youtu.be":
        return parsed_url.path[1:]
    if "youtube.com" in parsed_url.netloc and "v" in parse_qs(parsed_url.query):
        return parse_qs(parsed_url.query)["v"][0]
    return ""


def improve_transcript_punctuation(transcript: str) -> str:
    """Improve punctuation in transcript."""
    sentences = sent_tokenize(transcript)
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if (
            sentence
            and not sentence.endswith(("?", "!", '"', "'", "."))
            and sentence[-1] not in config.abbreviations
        ):
            sentence += "."
        processed_sentences.append(
            sentence[0].upper() + sentence[1:] if sentence else ""
        )
    return " ".join(processed_sentences)


# Streamlit App
def main():
    """Run the Streamlit app."""
    ensure_nltk_resources()
    st.set_page_config(layout="wide", page_title="Political Speech Analyzer")
    st.title("Political Speech Analyzer")

    with st.sidebar:
        st.header("Input")
        youtube_url = st.text_input("Enter YouTube URL:", value="")

        st.header("Analysis Options")
        selected_model_name = st.selectbox("Select Model:", list(MODELS.keys()))
        st.write(MODELS[selected_model_name]["description"])
        chunk_size = st.slider(
            "Chunk Size (words):", 50, 500, Constants.DEFAULT_CHUNK_SIZE.value, 10
        )
        top_n_chunk = st.slider(
            "Top Keywords per Chunk:",
            1,
            20,
            Constants.DEFAULT_TOP_N_KEYWORDS_CHUNK.value,
            1,
        )
        top_n_overall = st.slider(
            "Top Keywords Overall:",
            1,
            20,
            Constants.DEFAULT_TOP_N_KEYWORDS_OVERALL.value,
            1,
        )
        do_stemming = st.checkbox("Stemming", value=True)
        do_lemmatization = st.checkbox("Lemmatization", value=True)
        if do_stemming and do_lemmatization:
            st.info(
                "Both stemming and lemmatization selected. Lemmatization will be prioritized."
            )

    if st.button("Analyze Speech"):
        if not youtube_url.strip():
            st.error("Please enter a YouTube URL.")
        else:
            with st.spinner("Fetching and analyzing transcript..."):
                speech_text = get_youtube_transcript(youtube_url)
                if speech_text:
                    df = analyze_speech(
                        speech_text,
                        selected_model_name,
                        chunk_size,
                        top_n_chunk,
                        top_n_overall,
                    )
                    display_results(df, selected_model_name)

    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def display_results(df: pd.DataFrame, selected_model_name: str):
    """Display analysis results in Streamlit."""
    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.header("Speech Summary")
        if not df.empty:
            summary = compute_summary(df, selected_model_name)
            summary_df = pd.DataFrame.from_dict(
                summary, orient="index", columns=["Value"]
            )
            st.dataframe(
                summary_df.style.format(na_rep="N/A"), use_container_width=True
            )

        st.header(
            f"Overall {MODELS[selected_model_name]['task'].replace('_', ' ').title()} Distribution"
        )
        if not df.empty:
            output_counts = df["output"].value_counts().reset_index()
            output_counts.columns = ["Output", "Count"]
            color_map = {
                "bias": Constants.BIAS_COLORS.value,
                "hate_speech": Constants.HATE_COLORS.value,
                "sentiment": Constants.SENTIMENT_COLORS.value,
                "sarcasm": px.colors.qualitative.Plotly,
            }
            fig_output = px.bar(
                output_counts,
                x="Output",
                y="Count",
                color="Output",
                color_discrete_sequence=color_map.get(
                    MODELS[selected_model_name]["task"]
                ),
            )
            fig_output.update_layout(showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_output, use_container_width=True)

        st.header("Pronoun Usage")
        if not df.empty:
            pronoun_sums = {
                k: df["pronoun_counts"].apply(lambda x: x[k]).sum()
                for k in df["pronoun_counts"].iloc[0].keys()
            }
            pronoun_df = pd.DataFrame.from_dict(
                pronoun_sums, orient="index", columns=["Count"]
            ).reset_index()
            pronoun_df.columns = ["Pronoun Type", "Count"]
            fig_pronoun, ax = plt.subplots()
            sns.barplot(x="Pronoun Type", y="Count", data=pronoun_df, ax=ax)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig_pronoun, use_container_width=True)

    with col2:
        st.header("Complete Chunk Analysis")
        if not df.empty:
            st.dataframe(
                df.style.format({"output_confidence": "{:.3f}"}),
                use_container_width=True,
            )

    with st.expander("Overall Sentiment Distribution"):
        if not df.empty and "sentiment" in df.columns:
            st.write(
                df["sentiment"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )


def compute_summary(df: pd.DataFrame, selected_model_name: str) -> Dict[str, str]:
    """Compute summary statistics for display."""
    task = MODELS[selected_model_name]["task"]
    bias_mapping = {"left": -1, "center": 0, "right": 1}
    sentiment_mapping = {
        "Very Negative": -1,
        "Negative": -0.5,
        "Neutral": 0,
        "Positive": 0.5,
        "Very Positive": 1,
    }

    weighted_bias_score = (
        (df["output"].map(bias_mapping) * df["output_confidence"]).mean()
        if task == "bias"
        else 0
    )
    if task == "sentiment":
        df["numeric_sentiment"] = df["sentiment"].map(sentiment_mapping)
        df["chunk_length"] = df["text"].apply(lambda x: len(x.split()))
        total_length = df["chunk_length"].sum()
        weighted_sentiment_score = (
            df["numeric_sentiment"] * (df["chunk_length"] / total_length)
        ).sum()
    else:
        weighted_sentiment_score = 0

    combined_score = 0.5 * weighted_bias_score + 0.5 * weighted_sentiment_score
    return {
        "Overall Sentiment": (
            df["sentiment"].mode()[0] if "sentiment" in df.columns else "N/A"
        ),
        "Overall Bias/Hate/Sarcasm": df["output"].mode()[0],
        "Average Output Confidence": f"{df['output_confidence'].mean():.2f}",
        "Top Keywords": ", ".join(df["overall_keywords"].iloc[0]),
        "Pronoun Usage": ", ".join(
            f"{key}: {value}"
            for key, value in {
                k: df["pronoun_counts"].apply(lambda x: x[k]).sum()
                for k in df["pronoun_counts"].iloc[0].keys()
            }.items()
        ),
        "Unified Score": f"{combined_score:.2f}",
    }


if __name__ == "__main__":
    main()
