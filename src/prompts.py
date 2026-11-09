TRIPLE_PROMPT = (
    "Extract the factual relationships stated in the passage below from the 2038 Novel "
    "Reply with a JSON list of objects, each holding exactly the "
    'keys "subject", "relation" and "object". Use the wording of the passage. '
    "Reply with an empty list if it states no relationships.\n\n"
    "Passage:\n{text}\n\n"
    "JSON:"
)