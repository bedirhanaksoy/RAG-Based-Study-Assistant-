def prompt_formatter(tokenizer, query, context_items):
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer. If the answer is not contained within the context, please answer based on your own knowledge without mentioning the lack of context.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What is the primary function of the paging algorithm in a virtual memory system??
Answer: The paging algorithm in a virtual memory system is based on the working-set concept, and it’s designed to quickly map virtual addresses to physical addresses. The operating system maintains several page lists to optimize memory utilization and performs fast mapping.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query. Use the related context informations to provide a detailed and accurate answer without giving excuses about lack of context.:
{context}
\nUser query: {query}
Answer:"""

    filled = base_prompt.format(context=context, query=query)

    dialogue = [{"role": "user", "content": filled}]

    prompt = tokenizer.apply_chat_template(
        conversation=dialogue,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt
