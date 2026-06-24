import torch


def get_word_embedding(
        sentence,
        target_word,
        tokenizer,
        model
):

    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    hidden_states = outputs.hidden_states

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    target_positions = []

    for i, token in enumerate(tokens):

        clean_token = token.replace("##", "")
        clean_token = clean_token.replace("Ġ", "")

        if clean_token.lower() == target_word.lower():

            target_positions.append(i)

    if len(target_positions) == 0:

        return None

    position = target_positions[0]

    embeddings = []

    for layer in hidden_states:

        embeddings.append(
            layer[0, position]
            .cpu()
            .numpy()
        )

    return embeddings