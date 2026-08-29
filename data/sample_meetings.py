"""
Synthetic casual meeting transcripts for the Sticky Note Generation project.
Each meeting is a list of turns: (timestamp_sec, speaker, text)
Each meeting also has a human-written 'reference_notes' list used as gold
summaries for ROUGE/BLEU/METEOR evaluation, and 'gold_important_idx' -
indices of turns a human annotator marked as "sticky-note worthy" (used
for precision/recall/F1 of the importance classifier).
"""

MEETINGS = {

    "project_standup": {
        "turns": [
            (0,  "Aditi",  "Morning everyone, hope the weekend was good."),
            (4,  "Ravi",   "Yeah, pretty chill. Watched a movie."),
            (8,  "Aditi",  "Nice. Okay let's get started, we have a lot to cover today."),
            (12, "Ravi",   "So I finished the data cleaning for the churn dataset yesterday."),
            (18, "Meera",  "Oh nice, did you handle the missing values in the tenure column?"),
            (24, "Ravi",   "Yes, I imputed them using median tenure per contract type."),
            (30, "Aditi",  "Good. Can you push that to the repo by end of day?"),
            (34, "Ravi",   "Sure, I'll push it to the telco-churn-revenue-at-risk repo by 6pm today."),
            (40, "Meera",  "I'm still stuck on the segmentation queries, the SQL is throwing an error."),
            (46, "Aditi",  "What's the error exactly?"),
            (49, "Meera",  "Something about a group by clause not matching the select columns."),
            (55, "Ravi",   "Ah I had that too, you probably need to add customer_id to the group by."),
            (61, "Meera",  "Let me try that."),
            (63, "Aditi",  "Okay, let's decide - we will finalize the five business questions by Thursday."),
            (70, "Meera",  "Agreed, Thursday works."),
            (73, "Ravi",   "Agreed."),
            (75, "Aditi",  "Great, so action items: Ravi pushes cleaned data today, Meera fixes segmentation query "
                            "by tomorrow, and I will draft the five business questions by Wednesday for review."),
            (85, "Meera",  "Sounds good."),
            (87, "Ravi",   "One more thing, the client call got moved to Friday 3pm instead of Thursday."),
            (93, "Aditi",  "Okay noted, Friday 3pm for the client call."),
            (97, "Meera",  "Anything else? Otherwise I have another meeting in five minutes."),
            (102,"Aditi",  "That's all for today, thanks everyone."),
        ],
        "gold_important_idx": [3, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19],
        "reference_notes": [
            "Ravi finished data cleaning for churn dataset, imputed missing tenure values using median per contract type.",
            "Ravi to push cleaned data to telco-churn-revenue-at-risk repo by 6pm today.",
            "Meera facing SQL error in segmentation query; fix is adding customer_id to group by clause.",
            "Team agreed to finalize the five business questions by Thursday.",
            "Action items: Ravi - push data today; Meera - fix segmentation query by tomorrow; Aditi - draft five business questions by Wednesday.",
            "Client call rescheduled from Thursday to Friday 3pm."
        ]
    },

    "faculty_curriculum_meeting": {
        "turns": [
            (0,  "Dr. Rao",   "Thanks for joining, let's discuss the MDA504 syllabus revision."),
            (6,  "Dr. Iyer",  "Sure, I think we need more emphasis on ensemble methods this semester."),
            (12, "Dr. Rao",   "Agreed, students struggled with that unit in the last exam."),
            (18, "Prof. Nair","Also can we push the ASR assignment deadline? Students requested more time."),
            (25, "Dr. Rao",   "How much more time are we talking about?"),
            (28, "Prof. Nair","Maybe one week, so from the 20th to the 27th."),
            (34, "Dr. Iyer",  "I'm fine with that, as long as it doesn't clash with the ensemble assignment."),
            (40, "Dr. Rao",   "It doesn't, that one is due on the 15th."),
            (44, "Prof. Nair","Okay then let's move ASR deadline to the 27th."),
            (48, "Dr. Rao",   "Decision made - ASR assignment deadline moves to the 27th."),
            (53, "Dr. Iyer",  "Should we also add a practice quiz before the unit 4 exam?"),
            (58, "Dr. Rao",   "Good idea. Prof. Nair can you prepare 20 MCQs by next Monday?"),
            (65, "Prof. Nair","Yes, I'll have the MCQs ready by Monday."),
            (69, "Dr. Iyer",  "What topics should the MCQs cover?"),
            (72, "Dr. Rao",   "Multilayer perceptrons and ensemble methods, matching the exam style."),
            (78, "Prof. Nair","Got it, I'll base them on last year's question paper format."),
            (83, "Dr. Rao",   "Perfect. Let's also remind students the exam is on the 30th."),
            (88, "Dr. Iyer",  "I'll send that reminder on the class group today."),
            (93, "Dr. Rao",   "Great, thanks everyone, meeting adjourned."),
        ],
        "gold_important_idx": [1, 3, 5, 8, 9, 11, 12, 14, 15, 16, 17],
        "reference_notes": [
            "Decision: more emphasis needed on ensemble methods this semester after students struggled last exam.",
            "ASR assignment deadline moved from the 20th to the 27th.",
            "Ensemble assignment deadline remains the 15th, no clash with ASR extension.",
            "Prof. Nair to prepare 20 MCQs on multilayer perceptrons and ensemble methods by Monday, based on last year's paper format.",
            "Unit 4 exam scheduled for the 30th; Dr. Iyer to remind students via class group today."
        ]
    },

    "casual_team_coffee_chat": {
        "turns": [
            (0,  "Sam",   "Hey, grabbing coffee, want one?"),
            (4,  "Priya", "Sure, black coffee please."),
            (8,  "Sam",   "So how's the Accenture onboarding going?"),
            (12, "Priya", "Pretty good, though the training cohort schedule is intense."),
            (18, "Sam",   "Yeah I heard. Are you still on the churn project?"),
            (22, "Priya", "Yes, we're presenting the retention recommendations next Tuesday."),
            (28, "Sam",   "Oh that's soon. Need any help with the slides?"),
            (32, "Priya", "Actually yes, could you review my deck before Monday?"),
            (37, "Sam",   "Sure, send it over tonight and I'll review by tomorrow morning."),
            (43, "Priya", "Thanks, appreciate it."),
            (45, "Sam",   "By the way did you sign up for the AI hackathon?"),
            (49, "Priya", "Not yet, when's the deadline?"),
            (52, "Sam",   "This Friday I think, let me check... yes, Friday midnight."),
            (58, "Priya", "Okay I'll register today then."),
            (61, "Sam",   "Cool. Anyway, back to work, catch you later."),
            (64, "Priya", "See ya."),
        ],
        "gold_important_idx": [5, 7, 8, 12, 13],
        "reference_notes": [
            "Priya presenting retention recommendations for churn project next Tuesday.",
            "Sam to review Priya's slide deck by tomorrow morning after she sends it tonight.",
            "AI hackathon registration deadline is this Friday midnight; Priya to register today."
        ]
    },
}
