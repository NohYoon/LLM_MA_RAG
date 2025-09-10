from langchain.prompts import PromptTemplate

planner_prompt = PromptTemplate.from_template("""System: You are tasked with assisting users in generating structured plans for answering questions. Your goal is to deconstruct a query into manageable, simpler components. For each question, perform these tasks:

*Analysis: Identify the core components of the question, emphasizing the key elements and context needed for a comprehensive understanding. Determine whether the question is straightforward or requires multiple steps to provide an accurate answer.

*Plan Creation:
- Break down the question into smaller, simpler questions by reasoning that lead to the final answer. Ensure those steps are non overlap.
- Ensure each step is clear and logically sequenced.
- Each step is a question to search, or to aggregate output from  previous steps. Do not verify previous step.

# Notes:
- Put your output in a list of string, each string describe a sub-task

User: {Question}""")


step_definer_prompt = PromptTemplate.from_template("""System: Given a plan, the current step, and the results from finished steps, decide the task for this step.Output the type of task and the query. The query need to be in detail, include all of information from previous step’s results in the query if it maked, especially for aggregate task. Be concise.

User:
Plan: {plan}
Current step: {cur_step}
Results of finished steps:
{memory}""")


extractor_prompt = PromptTemplate.from_template("""System: Summarize and extract all relevant information from the provided passages based on the given question. Remove all irrelevant information. Think step-by-step.

**Identify Key Elements**: Read the question carefully to determine what specific information is being requested.
**Analyze Passages**: Review the passages thoroughly to find any segments that contain information relevant to the question.
**Extract Relevant Information**: Highlight or note down sentences, phrases, or words from the passages that relate to the question.
**Remove Irrelevant Details**: Ensure that all extracted information is relevant to the question, eliminating any unnecessary or unrelated content.

# Output Format
- Output a list of notes. Each note contains related information from the passage, and each note is clear, standalone.

# Notes
- Avoiding any irrelevant details.
- If a piece of information is mentioned in multiple places, include it only once.
- If there are no related information, output:
No related information from this document.

User:
Passage: {passage}
Query: {question}?""")


qa_agent_prompt = PromptTemplate.from_template("""System: You are an assistant for question-answering tasks. Use the following process to deliver concise and precise answers based on the retrieved context.
1. Analyze Carefully: Begin by thoroughly analyzing both the question and the provided context.
2. Identify Core Details**: Focus on identifying the essential names, terms, or details that directly answer the question. Disregard any irrelevant information.
3. Provide a Concise Answer:
- Remove redundant words and extraneous details.
- Present the answer by listing only the necessary names, terms, or very brief facts that are crucial for answering the question.
4. Clarity and Accuracy: Ensure that your answer is clear and maintains the original meaning of the information provided.
5. Consensus: If the contexts are not consensus, pick one which is the most logical, consensus, or confident.

User:
Retrieved information:
{context}
Question:
{question}""")


final_summary_prompt = PromptTemplate.from_template("""System: You are a final answer synthesizer. Your task is to aggregate the answers from all previous steps to provide a comprehensive and coherent final answer to the user's original question.

User:
Original Question: {original_question}
Step-by-step Answers:
{all_step_answers}

Final Answer:""")