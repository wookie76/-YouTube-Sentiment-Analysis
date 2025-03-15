
# YouTube Sentiment Analysis & Transcript Analyzer with Hugging Face Transformers, spaCy, and Streamlit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

**Unlock the insights hidden within YouTube video transcripts!** This powerful, yet user-friendly,
project combines state-of-the-art Natural Language Processing (NLP) techniques to perform
in-depth sentiment analysis, keyword extraction, and summarization of YouTube videos.  Whether
you're a researcher, marketer, content creator, or just curious about the underlying tone and
topics of a video, this tool provides valuable insights.

## Overview

This project leverages the power of Hugging Face Transformers, spaCy, KeyBERT/Textrank, and
YouTube Transcript API to transform raw YouTube transcripts into actionable data. It's
designed to be highly configurable and adaptable to videos of varying lengths, from short
clips to long-form lectures. The interactive Streamlit interface makes it easy to explore
the results, visualizing sentiment, bias, hate speech, and sarcasm detection (using
pre-trained models), along with pronoun usage and overall key themes.

**Key Features:**

*   **Sentiment Analysis:** Go beyond simple positive/negative and understand the nuances
    of sentiment across a video, using a five-point scale (Very Negative to Very Positive).
    Detects overall video setiment, and segment by segment.
*   **Multi-faceted Analysis:**  Not only gauge general *sentiment*, but delve into specific
    aspects, choosing from various lenses:
      -   **Political Bias Detection:** Identify left, center, or right-leaning viewpoints
          using the `bucketresearch/politicalBiasBERT` model.
      -    **Detecting Sarcasm:** See portions of dialog containing likely sarcastic content
      -   **Hate Speech Detection:** Flag potentially harmful content with
          the `facebook/roberta-hate-speech-dynabench-r4-target` model.
      -   **Overall sentiment scoring** Compute overall metrics quantifying content
          sentiment, along five point spectrum, from very negative, to very positive.
*   **Keyword & Keyphrase Extraction:**  Uncover the most important topics and themes using
     a combination of TextRank and, optionally, KeyBERT (for enhanced keyphrase
    extraction – see installation instructions below). Configurable parameters (see User
    Guide below).
*   **Dynamic Chunking:**  The transcript is intelligently divided into chunks for
    analysis. Adjust chunk size based on video length for optimal granularity – from
    fine-grained sentence-level analysis to broader topic-level overviews.  Sentence
    boundaries are prioritized for semantically meaningful chunks.
*   **Pronoun Usage Analysis:** Gain insights into the speaker's perspective by
    analyzing the frequency of different pronoun categories (first-person singular,
    third-person plural, etc.).
*   **Interactive Visualization:** Explore the results with interactive charts and
    tables powered by Streamlit and Plotly. Visualize sentiment distribution, pronoun
    usage, and keyword frequencies.
*   **Transcript Summarization:**  Generate concise summaries of the video using the
    `facebook/bart-large-cnn` summarization pipeline. Handle even very long videos by
    automatically truncating transcripts to the maximum token length, with informative
    notifications.
*   **Easy-to-Use Interface:**  The Streamlit web application provides a clean,
    intuitive interface for entering a YouTube URL and adjusting analysis parameters. No
    coding required!
*   **Robust Error Handling:** Gracefully handles common issues like unavailable
    transcripts, disabled transcripts, and invalid video IDs.
*    **Highly extensible** Models, defaults easily customized, adaptable to many additional use
 cases beyond YouTube analysis.

## Why This Project?

*   **Research:** Analyze political discourse, track public opinion, study
    communication patterns.
*   **Marketing & Content Creation:** Understand audience reaction, identify key
    messages, optimize content strategy.
*   **Social Listening:** Monitor brand perception, track trends, detect potential
    crises.
*   **Education:** Analyze lectures and presentations, identify key takeaways,
    facilitate understanding.

## Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/<your-username>/YT-Hugging-Sentiment.git
    cd YT-Hugging-Sentiment
    ```

2.  **Install dependencies:**

    ```bash
    pip install .
    ```
    *Tip: For optimal compatibility with the YouTube Transcript API, version 0.6.3 has
    been found to work reliably in the current development environment. While newer
    versions are available, you may encounter issues. If you experience problems, consider
     specifically installing this version:*

    ```bash
     pip install youtube-transcript-api==0.6.3
    ```
     *Optional* spaCy Model
     If error run this command in your environment, from terminal:
        `python -m spacy download en_core_web_sm`

    To install with KeyBERT support (optional):

    ```bash
     pip install .[keybert]
    ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run YT-Hugging-Sentiment.py
    ```

4.  **Enter a YouTube URL and explore!** *Note: The video must have English subtitles
    available for the transcript to be retrieved.*

## Development Environment

This project was developed using the following environment:

*   **WSL (Windows Subsystem for Linux):**  Provides a Linux environment within Windows.
*   **Miniconda:**  A lightweight package and environment manager.
*   **Conda Environment (Python 3.12):**  A dedicated environment ensuring consistent
    dependencies. While the project is compatible with Python 3.8+, using a dedicated
    environment with the specified dependencies in `pyproject.toml` is highly
    recommended for optimal reproducibility.

## Configuration Guide (In-App)
The Streamlit interface offers interactive sliders to refine results

**I. Video Length Categories:**

*   **Short (< 5 minutes):** News, tutorials.
*   **Medium (5-30 minutes):** Vlogs, explainers.
*   **Long (30+ minutes):** Lectures, podcasts.

**II. Sliders:**

1.  **Chunk Size:** Divide transcript for analysis.
    *   Short: 50-100 words.
    *   Medium: 100-200 words.
    *   Long: 250-500 words.

2.  **Top Keywords/Chunk:** Important words *per chunk*.
    *   Short: 3-5.
    *   Medium: 5-7.
    *   Long: 7-10.

3.  **Top Keywords Overall:** Important words, *whole video*.
    *   Short: 5-10.
    *   Medium: 10-20.
    *   Long: 20-50.

4.  **Lemmatization**: consider just root meanings, or verb tenses,
    plurals separately

## A Note on Perspective

This project was developed by an English-speaking, U.S.-centric, heteronormative white male
developer.  Like all developers, I have my own inherent biases and blind spots. I've strived
to create a tool that is as objective and inclusive as possible, leveraging established NLP
models.  However, it's important to remember that the analysis is ultimately based on the
available data (the transcript and the chosen models) and may reflect limitations in that data
or in the models themselves.  I encourage users to consider these factors when interpreting
the results. User feedback and contributions are always appreciated, for increasing use and
adaptablility of code to wide range of projects and languages.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

*   Hugging Face Transformers
*   spaCy
*   Streamlit
*   YouTube Transcript API
*   KeyBERT (optional dependency)
* TextRank
