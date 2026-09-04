# 📝 AI-Based Sticky Note Generator

An NLP-based application that automatically converts conversational transcripts into concise and meaningful **sticky notes** by identifying the most important information from a conversation.

The project combines **Natural Language Processing, text summarization, feature extraction, machine learning, and explainability** to generate useful notes from conversations.

---

## 📌 Project Objective

The main objective of this project is to develop an AI system that can:

* Identify important information from conversations.
* Extract the most relevant sentences or content.
* Generate concise sticky notes automatically.
* Compare different approaches for sticky-note generation.
* Evaluate the quality of generated notes using suitable metrics.
* Provide an interactive Streamlit dashboard for users.

---

## 🎯 Problem Statement

Long conversations, meetings, interviews, and discussions often contain a large amount of information. Manually identifying and writing down the important points can be time-consuming.

This project addresses this problem by automatically processing a conversation and generating **short, meaningful sticky notes containing the key information**.

---

## 🏗️ System Workflow

```text
Conversation / Audio
        ↓
Speech-to-Text / Transcript
        ↓
Text Preprocessing
        ↓
Feature Extraction
        ↓
Important Information Identification
        ↓
Sticky Note Generation
        ↓
Evaluation
        ↓
Streamlit Dashboard
```

---

## 🔍 Key Features

### 1. Transcript Input

The system allows users to provide a conversation transcript as input.

The Streamlit dashboard also provides an option to **upload an audio file**, which can be converted into text before generating sticky notes.

### 2. Sample Conversations

The dashboard provides sample conversations so that users can test the application without preparing their own input.

### 3. Important Information Identification

The system analyses the conversation and identifies information that is useful for creating concise notes.

The approach considers textual characteristics and extracted features to determine which parts of the conversation are more important.

### 4. Sticky Note Generation

The identified important information is transformed into short and readable sticky notes.

The objective is to retain the main meaning while removing unnecessary conversational content.

### 5. Model Evaluation

The generated sticky notes are evaluated to understand how effectively the system identifies important information.

Evaluation helps compare the generated output against the expected/reference information.

### 6. Interactive Streamlit Dashboard

The project includes an interactive dashboard where users can:

* Upload their own transcript.
* Upload audio.
* Select sample conversations.
* Generate sticky notes.
* View the generated output.
* Explore model/evaluation results.

---

## 🧠 Features Used

The model uses textual and linguistic characteristics extracted from the conversation.

Examples include:

* Sentence length
* Word frequency
* Position of the sentence
* Keyword/important-word information
* Text-based relevance features

These features help the model distinguish potentially important information from less relevant conversational content.

---

## 🤖 Machine Learning Approach

The project follows a supervised machine-learning approach for identifying important information from the transcript.

The overall process includes:

1. Preparing the conversation data.
2. Cleaning and preprocessing the text.
3. Splitting the text into sentences.
4. Extracting relevant features.
5. Training the machine-learning model.
6. Predicting the importance of sentences.
7. Selecting important information.
8. Generating the final sticky notes.

---

## 📊 Evaluation

The system is evaluated using appropriate NLP/model-performance measures to determine how effectively important information is identified and how useful the generated sticky notes are.

The evaluation focuses on:

* Quality of important-information identification.
* Agreement between generated and expected information.
* Overall usefulness and conciseness of the generated sticky notes.

The results are presented through the project dashboard and supporting analysis.

---

## 📈 What Contributes Most to Performance?

The feature analysis is used to understand which characteristics contribute most to identifying important information.

Features related to **sentence relevance, word importance, sentence position, and textual characteristics** help the model distinguish key information from less useful conversational content.

This analysis provides insight into why the model selects particular sentences for sticky-note generation.

---

## 🖥️ Streamlit Dashboard

The project includes a Streamlit-based user interface.

### Dashboard capabilities:

**Input Options**

* Upload transcript
* Upload audio
* Use sample conversations

**Processing**

* Text preprocessing
* Important information identification
* Sticky-note generation

**Output**

* Generated sticky notes
* Model/evaluation information
* Results for analysis

The dashboard makes the system easier to demonstrate and use without requiring users to run individual Python scripts.

---

## 📂 Project Structure

```text
sticky-notes-generation/
│
├── app/
│   └── dashboard.py
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── outputs/
│   └── ...
│
├── figures/
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

> Folder names may vary depending on the final version of the repository.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Natural Language Processing (NLP)**
* **Streamlit**
* **Matplotlib / Seaborn**
* **Jupyter Notebook**

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/rubina1905/sticky-notes-generation.git
cd sticky-notes-generation
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/dashboard.py
```

If `dashboard.py` is located in the root directory, use:

```bash
streamlit run dashboard.py
```

---

## 💡 Example Use Cases

The system can be applied to:

* Meeting summaries
* Interview conversations
* Customer interactions
* Classroom discussions
* Business discussions
* Personal voice notes
* Task/action-item extraction
* Conversation summarization

---

## ⚠️ Limitations

Although the system can identify and generate useful sticky notes, there are some limitations:

* The quality of the output depends on the quality of the input transcript.
* Informal conversations can be more difficult to process than structured speech.
* Ambiguous statements may not always be identified correctly.
* Background noise can affect audio transcription.
* Important information can sometimes be missed.
* Very long conversations may require additional processing or summarization.
* The generated sticky notes may not always perfectly represent the context of the original conversation.

---

## 🔮 Future Applications and Improvements

Future versions of the project could include:

* More advanced transformer-based NLP models.
* Improved contextual understanding using models such as BERT or other large language models.
* Automatic detection of action items and deadlines.
* Speaker identification and speaker-wise summaries.
* Multilingual sticky-note generation.
* Better handling of noisy audio.
* Real-time sticky-note generation during meetings.
* Integration with meeting and productivity applications.
* Personalized sticky notes based on user requirements.
* Improved explainability to show why particular information was selected.

---

## 🎯 Conclusion

The **AI-Based Sticky Note Generator** demonstrates how NLP and machine learning can be used to convert unstructured conversations into concise and useful information.

The project focuses not only on generating sticky notes but also on understanding **how effectively important information is identified and which features contribute to model performance**.

The Streamlit dashboard provides an interactive way to test the system using transcripts, audio files, and sample conversations, making the solution practical and easy to demonstrate.

---

## 👩‍💻 Author

**Janani- 2582414**

**Rubina- 2582444**


MSc Data Analytics
Christ University
