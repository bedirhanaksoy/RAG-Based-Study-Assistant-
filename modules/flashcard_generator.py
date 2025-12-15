"""Flashcard generation using RAG pipeline."""

from modules.embedder import retrieve_relevant_resources
from modules.prompt_builder import prompt_formatter
from modules.llm_inference import generate_answer
import json


# Question types for diversity - cycled through round-robin style
QUESTION_TYPES = [
    "DEFINITION",
    "PURPOSE",
    "HOW IT WORKS",
    "PROCESS OR STEPS",
    "BENEFITS OR ADVANTAGES",
    "RELATIONSHIPS",
    "REQUIREMENTS",
    "COMPARISON",
    "DESIGN REASONING",
    "SCOPE OR BOUNDARIES",
    "LIMITATIONS OR CONSTRAINTS",
    "FAILURE OR NEGATIVE OUTCOMES",
    "RISKS",
    "EFFICIENCY OR EFFECTIVENESS",
    "COORDINATION OR TIMING",
    "LIFECYCLE OR STATE CHANGES",
    "EXAMPLE OR APPLICATION",
    "EDGE OR EXCEPTIONAL CASES"
]

# Persistent counter to track position in QUESTION_TYPES across requests
_question_type_index = 0


def _strip_context_phrase(text: str) -> str:
    """Remove boilerplate like 'according to the context' from model outputs."""
    phrases = [
        "according to the provided context",
        "according to the context",
        "based on the provided context",
        "based on the context",
        "from the provided context",
        "from the context",
    ]
    for p in phrases:
        text = text.replace(p, "").replace(p.capitalize(), "")
    # Remove the word "primary" (case-insensitive)
    text = text.replace("primary ", "").replace("Primary ", "")
    # Truncate at first question mark (remove any trailing text after the question)
    if "?" in text:
        text = text[:text.index("?") + 1]
    # Collapse extra whitespace
    return " ".join(text.split())


def generate_questions(topic, embedding_model, embeddings_tensor, pages_and_chunks, 
                      tokenizer, llm_model, question_count=5):
    """Generate flashcard questions based on a topic, one at a time.
    
    Args:
        topic: The topic to generate questions about
        embedding_model: SentenceTransformer model
        embeddings_tensor: Embedded vectors for all chunks
        pages_and_chunks: List of chunk metadata
        tokenizer: LLM tokenizer
        llm_model: LLM model
        question_count: Number of questions to generate (max 5)
        
    Returns:
        list: List of generated questions
    """
    # Limit to max 5 questions
    question_count = min(question_count, 5)
    
    # Retrieve 2x question_count relevant chunks for diversity
    n_resources = min(question_count * 2, len(pages_and_chunks))
    scores, indices = retrieve_relevant_resources(
        query=topic,
        embeddings=embeddings_tensor,
        model=embedding_model,
        n_resources_to_return=n_resources
    )
    
    context_items = [pages_and_chunks[i] for i in indices]
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()
    
    # Split contexts into question_count groups for diversity
    # Each question gets a different subset of contexts
    chunk_size = max(1, len(context_items) // question_count)
    context_groups = []
    for i in range(question_count):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        if i == question_count - 1:  # Last group gets remaining items
            end_idx = len(context_items)
        group_items = context_items[start_idx:end_idx]
        if group_items:
            context_groups.append(group_items)
    
    # Ensure we have enough context groups
    while len(context_groups) < question_count:
        context_groups.append(context_items[:chunk_size])
    
    questions = []
    
    # Access the global question type index for persistence across requests
    global _question_type_index
    
    # Generate one question per context group
    for q_num in range(question_count):
        if q_num >= len(context_groups):
            break
        
        current_context_items = context_groups[q_num]
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in current_context_items])
        
        # Build list of previously generated questions for the prompt
        prev_questions_text = "\n- ".join(questions) if questions else "None"
        
        # Select question type using persistent round-robin cycling through QUESTION_TYPES
        question_type = QUESTION_TYPES[_question_type_index % len(QUESTION_TYPES)]
        _question_type_index += 1
        
        question_prompt = f"""Based on the following context about "{topic}", generate exactly 1 question.

QUESTION TYPE (MUST FOLLOW EXACTLY):
- {question_type}
You must generate a question of this type about "{topic}". Do not generate a different type.

The question should be clear and concise.

IMPORTANT: Do NOT generate any similar questions to these previous questions (avoid the same framing, intent, or angle):
- {prev_questions_text}

DIVERSITY RULES (STRICT):
- The new question must focus on a different aspect/subtopic of "{topic}" than the previous questions.
- Avoid repeating the same “question framing” as recent questions (e.g., if prior questions ask for definition/purpose/role, do not produce another definition/purpose/role variant).
- Avoid vague filler words like "role", "primary", "purpose", "key aspect", "importance" unless {question_type} explicitly requires them.
- Do not use same phrases or wording as previous questions.

STANDALONE RULE:
The question must be fully understandable on its own and must NOT imply the existence of any text, passage, or external source.

DO NOT PROVIDE ANSWERS OR EXPLANATIONS.
Do not use phrases like "According to the context" or mention context at all.
Do not make any comments or evaluations; output only the question.

Generate 1 NEW question (single line, ending with a '?'):

Use these contexts to help you generate the question:
{context}

Generate 1 NEW question with the question type - {question_type} (single line, ending with a '?'):"""
        
        # DEBUG: Print the prompt
        print(f"\n{'='*60}")
        print(f"[DEBUG] Question {q_num + 1} Generation Prompt:")
        print(f"{'='*60}")
        print(question_prompt)
        print(f"{'='*60}\n")
        
        # Generate single question using LLM with retries
        max_retries = 3
        extracted = False
        for attempt in range(max_retries):
            question_text = generate_answer(
                tokenizer, llm_model, question_prompt,
                temperature=0.9,  # Slightly higher temp for diversity
                max_new_tokens=256
            )
            
            # Clean up the response: remove prompt, special tokens, and extra whitespace
            question_text = question_text.replace(question_prompt, "").strip()
            question_text = question_text.replace("<bos>", "").replace("<eos>", "").replace("<end_of_turn>", "").strip()
            question_text = question_text.replace("```", "").strip()
            question_text = _strip_context_phrase(question_text)
            
            # Extract first valid sentence/question
            # Split by common delimiters and take the first substantial line
            lines = question_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Remove leading numbering/bullets
                line = line.lstrip('0123456789.-) ')
                # Skip lines that are instructions/remnants from prompt
                if any(skip in line.lower() for skip in ["generate", "answer:", "--- ", "do not", "don't", "instruction", "context"]):
                    continue
                # Check if line has actual content (not just special chars or whitespace)
                if line and any(c.isalpha() for c in line) and len(line) > 5 and '?' in line:
                    # Remove any trailing artifacts like "--- " or "---"
                    line = line.split("---")[0].strip()
                    if line:
                        # Check for duplicates (case-insensitive)
                        line_lower = line.lower()
                        is_duplicate = any(q.lower() == line_lower for q in questions)
                        
                        if not is_duplicate:
                            questions.append(line)
                            extracted = True
                            break
            
            # If successfully extracted, break retry loop
            if extracted:
                break
        
        # If still not extracted after retries, add a generic question
        if not extracted:
            generic_q = f"What is a key aspect of {topic}?"
            if not any(q.lower() == generic_q.lower() for q in questions):
                questions.append(generic_q)
    
    return questions[:question_count]


def generate_flashcards(topic, file_name, question_count, embedding_model, 
                       embeddings_tensor, pages_and_chunks, tokenizer, llm_model, 
                       ask_function):
    """Generate complete flashcards (questions + answers) with context.
    
    Args:
        topic: The topic for flashcards
        file_name: The PDF file name (for reference)
        question_count: Number of flashcards to generate (max 5)
        embedding_model: SentenceTransformer model
        embeddings_tensor: Embedded vectors
        pages_and_chunks: Chunk metadata
        tokenizer: LLM tokenizer
        llm_model: LLM model
        ask_function: The ask() function from rag_pipeline
        
    Returns:
        dict: Contains 'topic', 'file_name', and 'flashcards' list
    """
    question_count = min(question_count, 5)
    
    # Generate questions
    questions = generate_questions(
        topic, embedding_model, embeddings_tensor, pages_and_chunks,
        tokenizer, llm_model, question_count
    )
    
    flashcards = []
    
    # For each question, generate an answer using the ask pipeline
    for question in questions:
        try:
            answer, context_items = ask_function(
                query=question,
                tokenizer=tokenizer,
                llm_model=llm_model,
                embedding_model=embedding_model,
                embeddings=embeddings_tensor,
                pages_and_chunks=pages_and_chunks,
                temperature=0.7,
                max_new_tokens=512,
                return_answer_only=False
            )
            
            # Clean answer from boilerplate
            answer = _strip_context_phrase(answer)
            
            # Clean context for JSON response
            cleaned_context = [
                {
                    "page_number": int(item["page_number"]) if isinstance(item["page_number"], (int, float)) else item["page_number"],
                    "sentence_chunk": item["sentence_chunk"]
                }
                for item in context_items
            ]
            
            flashcards.append({
                "question": question,
                "answer": answer,
                "context": cleaned_context
            })
        except Exception as e:
            # If answer generation fails, still include the question
            print(f"[WARN] Failed to generate answer for question: {question}. Error: {str(e)}")
            flashcards.append({
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "context": []
            })
    
    return {
        "topic": topic,
        "file_name": file_name,
        "question_count": len(flashcards),
        "flashcards": flashcards
    }
