# AI Prompt History — Student Success Copilot

**Tool Used:** Claude Code (Claude Opus 4.6 by Anthropic)
**Team Members:** 00020472, 00021430
**Date:** March 2026

AI tools were used in accordance with the module's AI usage policy to generate, debug, and improve code for this project.

---

## Planning & Brainstorming

1. "yo i need to build a hybrid ai system for my coursework, something that helps students figure out if theyre at risk of failing and gives them a study plan. it needs like 3 ai techniques minimum — thinking search algorithms, rule-based stuff, and ML"

2. "okay so lets call it Student Success Copilot. the user puts in their stress level, how many modules theyre doing, study hours, confidence etc and it spits out a risk level + personalised study schedule"

3. "can you set up the project structure? i want separate files for each ai component so its clean, and a streamlit frontend to tie it all together"

## Data & ML Model

4. "i need training data but obviously i dont have real student data lol. can you generate like 1000+ synthetic student records with realistic distributions? stress, confidence, workload, study hours, deadline proximity etc"

5. "aight now train a random forest on that data to predict risk level (low/medium/high). show me accuracy, precision, recall, f1, confusion matrix — the whole thing"

6. "how do i make the model fair across genders? also show me feature importance so i can explain which factors matter most"

7. "wait the model predictions dont match the rule engine sometimes — is that normal? how should i handle disagreements between the two?"

## Rule-Based Expert System

8. "build me a rule engine with forward chaining. i need like 20+ rules covering stress levels, confidence, workload, deadline urgency, study adequacy. it should derive new facts from the inputs step by step"

9. "add backward chaining too so it can explain WHY it gave a certain risk level — like trace back through the rules"

10. "the rules for study adequacy need to check the ratio of study hours to number of modules not just raw hours. if someone studies 10hrs but has 5 modules thats not enough"

11. "some rules arent firing properly — the combined risk rules need the individual assessments to run first. is the ordering right in forward chaining?"

## Search-Based Planner

12. "implement A* search and greedy best-first search for generating a weekly study schedule. the state is a partially filled timetable and the goal is all subjects allocated"

13. "the heuristic should factor in deadline urgency AND difficulty — harder subjects with closer deadlines should get scheduled first"

14. "can you compare both algorithms? like how many nodes each one explores, whether A* actually finds a better schedule, pros and cons"

15. "the search is taking forever with too many subjects — add a node limit so it doesnt hang"

## Fuzzy Logic

16. "add a fuzzy logic module as an optional 4th ai technique. use membership functions for stress and confidence to get a more nuanced risk score instead of hard cutoffs"

17. "make it so the app still works if scikit-fuzzy isnt installed — just skip that section gracefully"

## Streamlit UI

18. "create the streamlit app with a step-by-step flow: first collect student info, then subjects, then show all the results from each ai component"

19. "the app crashes when i submit the form — the helper functions are defined after theyre called. can you fix the ordering?"

20. "add tabs to compare A* vs greedy schedules side by side and an expander for the fired rules and backward chaining explanation"

21. "make it look clean with columns, metrics, charts for the ML probabilities and feature importance"

## Debugging & Testing

22. "fuzzy logic import fails even tho i pip installed scikit-fuzzy" — turned out networkx was a missing dependency

23. "check the whole app end to end for bugs. try edge cases like 1 module, max stress, deadline tomorrow etc"

24. "the schedule shows duplicate entries for the same subject on the same day — need to merge those into total hours per subject per day"

25. "backward chaining returns empty sometimes even when rules clearly fired. the issue was it wasnt recursing into derived facts properly"

## Deployment & Sharing

26. "create a github repo and push everything with a proper readme"

27. "i want anyone to be able to open this in google colab without installing anything. can you make a notebook version with all the code inline?"

28. "also make a second notebook that runs the actual streamlit app from colab using a tunnel so people can try the full ui"

29. "the tunnel is asking for a password thats so annoying. switch to something that just works"

30. "can you add an open in colab badge to the notebooks so its one click to open?"

---

## Revenue & Business Planning Session — March 26, 2026

31. "Please run AI_repo."

32. "I want to recall all the prompts I used — I forgot to screenshot them. Please recall all the prompts we have used, make them grammatically correct, and write them all in markdown file format."
