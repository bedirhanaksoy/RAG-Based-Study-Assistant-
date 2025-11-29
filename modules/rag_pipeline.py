from modules.embedder import retrieve_relevant_resources
from modules.prompt_builder import prompt_formatter
from modules.llm_inference import generate_answer


def ask(query, tokenizer, llm_model, embedding_model,
        embeddings, pages_and_chunks,
        temperature=0.7, max_new_tokens=512, return_answer_only=True):

    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model
    )

    context_items = [pages_and_chunks[i] for i in indices]

    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()

    prompt = prompt_formatter(tokenizer, query, context_items)

    output_text = generate_answer(
        tokenizer, llm_model, prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "")
    output_text = output_text.replace("Sure, here is the answer to the user query:\n\n", "")

    if return_answer_only:
        return output_text

    return output_text, context_items
