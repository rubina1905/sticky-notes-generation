# Sticky Note Generation from Casual Meeting Conversations

Extracts important content from meeting transcripts and generates concise sticky notes.

## Setup
```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','maxent_ne_chunker','maxent_ne_chunker_tab','words','stopwords']]"
```

## Run the CLI pipeline (trains model, evaluates, prints/saves sticky notes)
```bash
python train_and_evaluate.py
```
Outputs land in `outputs/report.json` (full metrics) and `outputs/sticky_notes.txt`.

## Run the dashboard
```bash
streamlit run app.py
```
Lets you pick a sample meeting or paste your own transcript (`Speaker: text` per line),
tune the importance threshold, and see sticky notes, the full feature table, and
model-evaluation metrics in one place.

## Project structure
```
data/sample_meetings.py   # 3 synthetic transcripts + gold labels + reference notes
features.py                # Task (i): 9 feature families -> per-turn feature vector
classifier.py               # Task (ii): RandomForest importance classifier, LOMO-CV
generate_notes.py           # Task (ii): segment merging + rule-based compression -> notes
evaluate.py                 # Task (iii): P/R/F1/Acc + ROUGE/BLEU/METEOR
train_and_evaluate.py        # end-to-end CLI runner
app.py                       # Streamlit dashboard
```

## Task (i): Features implemented
| Family | Concrete features |
|---|---|
| Keyword/keyphrase | YAKE keyphrase score per turn |
| Frequency of important terms | TF-IDF centrality vs. whole-meeting centroid |
| Speaker turns/interruptions | speaker-change flag, short-turn flag (backchannel proxy), long-turn flag |
| Repetition | max cosine similarity (TF-IDF) to any earlier turn |
| Q&A | question flag (`?` / question-word start), answer flag (follows a question) |
| Decisions/agreements | rule-based lexicon match ("agreed", "decided", "finalize"...) |
| Action-oriented | modal-verb + deadline lexicon match ("will", "by Friday", "action item"...) |
| Named entities | NLTK NE chunker (PERSON/ORG/GPE) + regex day/date detector |
| Contextual/positional | normalized position in meeting |

## Task (ii): Model
- **Selection model:** RandomForestClassifier (200 trees, depth 5, class-balanced) over the
  16-dim feature vector above, predicting P(turn is sticky-note worthy).
- **Segment merge:** adjacent important turns from the *same* speaker are merged into one
  segment so a single note isn't fragmented across consecutive utterances.
- **Compression:** rule-based - strips filler/discourse markers ("um", "yeah", "so", "okay"...),
  normalizes first-person pronouns to the speaker's name so each note reads standalone.
- Threshold (default 0.42) and top-K are both tunable in the dashboard.

## Task (iii): Evaluation results (on the 3 synthetic sample meetings)
**Classifier - Leave-One-Meeting-Out CV** (train on 2 meetings, test on the held-out one):

| Held-out meeting | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| project_standup | 0.625 | 0.455 | 0.526 | 0.591 |
| faculty_curriculum_meeting | 0.625 | 0.455 | 0.526 | 0.526 |
| casual_team_coffee_chat | 0.556 | 1.000 | 0.714 | 0.750 |
| **Overall** | **0.600** | **0.556** | **0.577** | **0.614** |

**Generated notes vs. human reference notes** (best-match alignment):

| Meeting | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|---|---|---|---|---|---|
| project_standup | 0.432 | 0.263 | 0.387 | 0.136 | 0.009 |
| faculty_curriculum_meeting | 0.496 | 0.304 | 0.452 | 0.188 | 0.000 |
| casual_team_coffee_chat | 0.422 | 0.201 | 0.377 | 0.067 | 0.000 |

Run `python train_and_evaluate.py` to regenerate these numbers along with feature importances.

## Task (iv): Analysis and Conclusion

**Effectiveness.** With only 3 annotated meetings (49 turns total) for training, the classifier
reaches overall F1 ≈ 0.58 under strict leave-one-meeting-out validation - i.e. tested on a
meeting type it never saw. It over-selects on the casual coffee-chat meeting (recall = 1.0,
lower precision) because informal turns still contain dates/action words that trigger the
model, and under-recalls on the more formal meetings where important content is phrased less
explicitly (e.g. terse agreements like "Agreed."). ROUGE-1 in the 0.42-0.50 range indicates the
generated notes capture roughly half the key unigrams a human would include - reasonable for a
lexicon+classifier system with no abstractive rewriting, but well below what an LLM-based
abstractive summarizer would achieve.

**Feature contribution.** TF-IDF centrality (how central a turn's vocabulary is to the whole
meeting) and repetition-to-earlier-turns rank highest, followed by positional feature and
keyword (YAKE) score. This makes intuitive sense: important statements (decisions, action items)
often restate or converge on the meeting's central topic and get referenced multiple times, and
important content clusters near the middle/end of segments (after context has been built and
before wrap-up). Lexicon-based flags (is_decision, has_person) rank lower individually because
they're sparse binary signals, but they matter most for the specific handful of decision/entity
turns.

**Limitations.**
- Trained on only 3 synthetic, English, text-only meetings - no real audio, disfluencies,
  overlapping speech, or code-switching.
- Rule-based compression is naive (pronoun swap + filler stripping); it doesn't rewrite
  syntax, so a few notes read slightly awkwardly (e.g. "Sam think" instead of "Sam thinks").
- No true interruption/overlap detection - speaker-change and short-turn flags are only proxies
  since transcripts here are strictly turn-taking, not diarized audio with overlap timestamps.
- Small labeled set limits how much the RandomForest can learn about rare-but-critical patterns
  (e.g. implicit decisions with no lexicon trigger word).

**Possible improvements / future applications.**
- Replace rule-based compression with an abstractive step (fine-tuned seq2seq or LLM prompt)
  conditioned on the same importance-classified segments, keeping the classifier as a
  precision filter.
- Add real audio features (prosody: pitch/energy emphasis, pause duration, speaking-rate change)
  once actual recordings are available - "changes in tone/emphasis" was in the assignment brief
  but requires audio, which the current text-only pipeline cannot capture.
- Use proper speaker diarization + timestamps to detect genuine interruptions/overlaps rather
  than the current turn-length proxy.
- Scale the labeled corpus (more meetings, multiple annotators, inter-annotator agreement) to
  get a more reliable supervised signal and enable held-out test-set evaluation instead of only
  cross-validation on 3 meetings.
- Applications: auto-generated meeting minutes for classroom/faculty meetings, standup-note
  bots for engineering teams, action-item trackers that sync directly to task boards.
