"""Default prompts."""

QA_TEMPLATE = """\
Please provide an answer solely on the provided context. Keep your answer \
concise and accurate. If question cannot be inferred from the context, \
just say that you don't know.
Context: {context}
Question: {question}
Answer: """
