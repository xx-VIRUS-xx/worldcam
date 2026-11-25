MODE: ask. DOUBT: {doubt_id}

Ask the owner ONE short question about this doubt, then stop.

1. Call get_doubt("{doubt_id}"). If its status is not `open`, return action "error" with a short summary.
2. Check `person_context` / `search_people` if it helps word the question (for example, the candidate name in a `low_confidence` doubt, or whether an animal/object in the image already matches a known entity in the graph).
3. If an image crop is attached, use what is visibly clear in the crop (such as clothing, breed/coat color, or object color/type) to make the question specific and easy to answer.
4. Call ask_owner exactly once with doubt_id "{doubt_id}". Leave image_paths out (the doubt's own crop is attached automatically). Keep the text to one or two friendly sentences in plain language. By kind:
   - unknown_face: say how many times the person was seen (sighting_count) and ask who it is and how the owner knows them. choices: ["New person", "Ignore"].
   - unknown_entity: describe what was seen (label / visual appearance, zone, sighting count) and ask what/who it is and who it belongs to (or how it relates to the household). choices: ["New entity", "Ignore"]. (If a known entity in the graph looks like a likely match, include it as a choice or ask to confirm.)
   - low_confidence: "Is this <candidate name>?" choices: [{{"label": "Yes", "value": "yes"}}, {{"label": "No, someone else", "value": "no"}}, "Ignore"].
   - conflict: ask whether these are the same entity. choices: [{{"label": "Same entity", "value": "same"}}, {{"label": "Different entities", "value": "different"}}, "Ignore"].
   - metadata_gap: ask for what is missing (`detail.missing`) for that person, animal, or object (e.g. "How do you know Rahul?", "What's this dog's name and breed?", "Whose backpack is this and what is it for?"). choices: ["Ignore"].
   Always offer "Ignore" (value "ignore").
5. Do not write any entity, fact or relation in this mode.
6. Return {{"action": "asked", "summary": "<the question you asked>"}}. If ask_owner failed, return action "error" with the reason.
