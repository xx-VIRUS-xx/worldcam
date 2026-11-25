# worldcam: vision (owner's intent, 2026-11-25)

**worldcam gives an autonomous agent eyes.** The agent watches the user's day-to-day
life through cameras, builds a **brain** (memory and context about people, animals,
objects and places), chats with the user, and **asks when it is unsure**. The user's
answer decides; the agent never settles an identity or a fact on its own.

The same core serves three applications, built in parallel as **profiles**:
| Profile | Intent |
|---|---|
| `cctv` | Smart CCTV: strangers, doors, packages, zones, a timeline of visitors, alerts |
| `companion` | AI companion: knows your people, pets, things and habits; talks about your day |
| `medical` | Caregiving / wellness: medication and meal routines, falls, inactivity, caregiver alerts |

A profile is configuration, not a second pipeline. Several profiles run at once on the
same cameras and the same brain; one observation can feed all of them.

## Principles
1. **Eyes**: turn video into structured observations (who/what, where, doing what, when) for people, animals and objects.
2. **Autonomous**: a long-running loop with event triggers and scheduled/random background loops (curiosity, look-around, consolidation, check-ins). All proactive messages share one attention budget per user.
3. **Chat**: two-way. Users ask ("who came today?"), teach ("this is my mother") and answer questions. Answers come from memory, cite their sources, and say "I don't know" rather than guess.
4. **Answer decides**: uncertainty becomes a doubt; a permitted user's answer settles it. Agent guesses are stored as unconfirmed with a confidence and a source.
5. **Roles**: the owner whitelists users and gives each a role (owner, household, caregiver, guest). Roles control who can chat, who gets which alerts, who may confirm what, and who sees images.
6. **Light**: stdlib + small local models where possible; cloud models (Claude, Gemini; optionally Jev for typed decisions) chosen explicitly per task.
7. **Local first, cloud by choice**: what leaves the machine (crops, text, to which provider) is explicit config per profile.

## Roadmap beyond this build
- **v2 voice**: the user talks to the agent and it talks back. Chat is channel-independent (`InboundMessage.modality`), so voice is another channel (wake word, speech-to-text, text-to-speech, speaker identification as another identity signal).
- **Later eyes**: V-JEPA-style video embeddings for activity and re-identification.

## Constraints to keep in view
- **Medical**: start as a wellness / caregiver aid with no diagnostic claims; measure accuracy before relying on any detector; the monitored person must consent.
- **Privacy**: a brain of daily life plus bystanders; retention windows, `forget`, provenance on every fact, a mic is more sensitive than a camera.
- **Autonomy limits**: the agent may observe, remember, ask, notify and chat. Contacting anyone outside the whitelist (caregiver, emergency services) needs an explicit per-profile rule.
- **Attention**: the agent is judged by how few, how valuable questions it asks.
