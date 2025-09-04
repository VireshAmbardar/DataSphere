import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

FILE_PROMPT ="""
You are a helpful assistant designed to provide insightful and clear responses based on user queries related to various document formats such as PDF, CSV, or DOC files. Your primary role is to analyze the context provided by relevant strings from these documents and answer questions in a descriptive and easily understandable manner.

Your task is to respond to user inquiries by first understanding the context derived from the provided messages, and then formulating clear and informative answers based on the user's specific question.

---

Please format your responses in a structured manner, ensuring clarity and coherence. Use bullet points or numbered lists where necessary to enhance readability.

---

When answering questions, keep the following in mind: 

- Ensure that your responses are thorough and provide detailed explanations where applicable.
- Maintain a friendly and professional tone, making the information accessible to users with varying levels of expertise.

---

Example of how you might structure an answer:

1. **Understanding the question:** [User's question here]
2. **Context analysis:** [Relevant string messages from the document]
3. **Answer:** [Your clear and descriptive response]

---

Be cautious about the following:

- Avoid jargon or overly technical language unless it's necessary for clarity.
- Make sure all responses are relevant to the user's query and based on the context provided.

Relevant Messages:
"""

def _build_user_message(user_query: str, context_string: str) -> str:
    # `context_snippets` is the string returned by _pretty_answer (a bunch of > quotes).
    return f"""
User Query: {user_query}

{FILE_PROMPT}
{context_string}
"""

#Response Generator
def response_generator(
    user_query: str,
    messages: str,  # your _pretty_answer output
    *,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stream: bool = True,
) -> str:
    system_msg = {"role": "system", "content": FILE_PROMPT.strip()}
    user_msg   = {"role": "user",   "content": _build_user_message(user_query, messages)}

    if not stream:
        chat = client.chat.completions.create(
            model=model,
            messages=[system_msg, user_msg],
            temperature=temperature,
            max_completion_tokens=max_tokens,  # Groq SDK example param name
        )
        return chat.choices[0].message.content

    # Streaming variant (prints as tokens arrive; returns full text at the end)
    chunks = client.chat.completions.create(
        model=model,
        messages=[system_msg, user_msg],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=True,  # stream like in Groq’s example
    )
    out = []
    for delta in chunks:
        piece = delta.choices[0].delta.content or ""
        print(piece, end="", flush=True)  # optional live echo
        out.append(piece)
    print()  # newline after stream
    return "".join(out)


