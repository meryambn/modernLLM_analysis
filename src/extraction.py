import json

from nltk.corpus import semcor


def extract_occurrences(
    target_word,
    max_examples=100
):
    """
    Extract sentences containing a target word
    from SemCor.
    """

    contexts = []

    for sentence in semcor.sents():

        words = [
            word.lower()
            for word in sentence
        ]

        if target_word.lower() in words:

            contexts.append(
                " ".join(sentence)
            )

        if len(contexts) >= max_examples:
            break

    return contexts


def save_contexts(
    contexts,
    filepath
):
    """
    Save contexts to JSON.
    """

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            contexts,
            f,
            indent=4
        )


def load_contexts(
    filepath
):
    """
    Load contexts from JSON.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        contexts = json.load(f)

    return contexts

def save_word_contexts(
    word,
    contexts
):

    filepath = f"../data/{word}_contexts.json"

    save_contexts(
        contexts,
        filepath
    )

def load_word_contexts(
    word
):

    filepath = f"../data/{word}_contexts.json"

    return load_contexts(
        filepath
    )