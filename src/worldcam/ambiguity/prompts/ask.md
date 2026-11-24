MODE: ask. DOUBT: {doubt_id}

Ask the owner ONE short question about this doubt, then stop.

1. Call get_doubt("{doubt_id}"). If its status is not `open`, return action "error" with a short summary.
2. Use person_context / search_people only if it helps word the question (for example, the name of the candidate person in a `low_confidence` doubt).
3. Call ask_owner exactly once with doubt_id "{doubt_id}". Leave image_paths out (the doubt's own crop is attached automatically). Keep the text to one or two friendly sentences in plain language. By kind:
   - unknown_face: say how many times the person was seen (sighting_count) and ask who it is and how the owner knows them. choices: ["New person", "Ignore"].
   - low_confidence: "Is this <candidate name>?" choices: [{{"label": "Yes", "value": "yes"}}, {{"label": "No, someone else", "value": "no"}}, "Ignore"].
   - conflict: ask whether these are the same person. choices: [{{"label": "Same person", "value": "same"}}, {{"label": "Different people", "value": "different"}}, "Ignore"].
   - metadata_gap: ask for what is missing (detail.missing), e.g. "How do you know Rahul?". choices: ["Ignore"].
   Always offer "Ignore" (value "ignore").
4. Do not write any person, fact or relation in this mode.
5. Return {{"action": "asked", "summary": "<the question you asked>"}}. If ask_owner failed, return action "error" with the reason.
