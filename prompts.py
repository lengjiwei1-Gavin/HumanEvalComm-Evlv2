# ==============================================================================
# PROMPT FOR THE ORIGINAL EVALUATOR (Baseline for Comparison)
# ==============================================================================
# This prompt is a faithful reproduction of the one used in the original paper.
# We will use this to run the baseline evaluation for direct comparison.

ORIGINAL_EVALUATOR_PROMPT_TEMPLATE = """The original description of a coding problem is modified so that the requirements become inconsistent, incomplete, or ambiguous. Given the modified description, some clarifying questions were raised to clarify the description. Given the original and modified problem description, evaluate the quality of the clarifying questions. Please provide an integer representing the quality of questions (3: Good questions that recover the modified requirements; 2: Fair questions but they cannot help recover the modified requirements; 1: No questions).

QUALITY=[your int]

Please also provide answers to the clarifying questions to recover the modified requirements in the original problem description compared to the modified one. If there are no clarifying questions at all, return empty answers.

ANSWERS='''{answer}'''

Please strictly follow the format QUALITY=[the int] and ANSWERS='''{answer}''' in the response! Surround your answer with markdown!

### Questions: {clarifying_questions}
### Modified Problem Description: {modified_problem}
### Original Description: {original_problem}
"""


# ==============================================================================
# PROMPT FOR THE CLASSIFIER (Our Improvement)
# ==============================================================================
# This prompt is for Module 1. It classifies the initial response.

# The prompt template we designed above
CLASSIFIER_PROMPT_TEMPLATE = """
You are an expert analyst responsible for classifying responses from a code generation Large Language model. Your task is to determine if a response is a "Clarifying Question" or a "Code Solution".

- A "Clarifying Question" is a response that asks for more information, points out ambiguities, or seeks to clarify the problem requirements without providing a functional code block.
- A "Code Solution" is a response that contains a markdown code block (```) intended to solve the problem, even if it is accompanied by explanatory text.

Respond with ONLY a JSON object in the format: {{"classification": "CATEGORY"}}, where CATEGORY is either "Clarifying Question" or "Code Solution".

Here are examples of how to classify:

---
### EXAMPLE 1 ###
[INPUT RESPONSE]:
To generate the Python code for the `candidate` function, I would need to understand the problem better. The function takes an integer `n` as input... However, the function does not specify what `k` is... To clarify, could you please provide more information about the problem you are trying to solve?

[CLASSIFICATION]:
{{"classification": "Clarifying Question"}}

---
### EXAMPLE 2 ###
[INPUT RESPONSE]:
Here is a possible implementation of the `candidate` function in Python 3:
```python
def candidate(paren_string: str) -> List[str]:
    groups = []
    current_group = ""
    for char in paren_string:
        if char == "(":
            current_group += char
        elif char == ")":
            groups.append(current_group)
            current_group = ""
        else:
            current_group += char
    return groups
```
This function uses a loop to iterate over the characters in the input string. It keeps track of the current group of parentheses using the `current_group` variable. When it encounters an opening parenthesis, it appends it to the current group. When it encounters a closing parenthesis, it appends the current group to the list of groups and resets the current group to an empty string. Finally, it returns the list of groups. Note that this implementation assumes that the input string is well-formed and that each group of parentheses is balanced. If the input string is not well-formed, the function may raise an exception or produce incorrect results. 

[CLASSIFICATION]:
{{"classification": "Code Solution"}}

---
### EXAMPLE 3 ###
[INPUT RESPONSE]:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j:
                if abs(numbers[i] - numbers[j]) > threshold:
                    return True
    return False
```

[CLASSIFICATION]: 
{{"classification": "Code Solution"}}

---
### YOUR TASK ###
Please classify the following response:

[INPUT RESPONSE]:
{code_llm_1st_response}

[CLASSIFICATION]:

"""

# ==============================================================================
# PROMPT FOR THE IMPROVED JURY EVALUATOR (Our New Method)
# ==============================================================================
# This prompt is for Module 2. It guides the multi-agent jury to provide a
# reliable, well-reasoned quality score. It incorporates Chain-of-Thought.

IMPROVED_EVALUATOR_PROMPT_TEMPLATE = """You are a meticulous quality evaluator. Your task is to evaluate the quality of a clarifying question on a scale of 1 to 3.

---
## KEY CONCEPTS ##
The modified problem will have one of these issues:
- **Ambiguity**: Multiple valid interpretations (e.g., "sort the list" without specifying ascending/descending)
- **Inconsistency**: Contradictory requirements (e.g., conflicting examples)
- **Incompleteness**: Missing critical information (e.g., "increment elements" without specifying by how much)

---
## EVALUATION CRITERIA ##
- **Score 3 (Good)**: Question precisely identifies the core issue and asks for the specific missing/unclear information
- **Score 2 (Fair)**: Question is relevant but too general or addresses a secondary issue instead of the main problem
- **Score 1 (Bad)**: Question is irrelevant, or model provided code instead of asking a question

---
### Negative Indicators (Should result in lower scores) ###
- **Vague Inquiries**: Questions that are too broad and do not pinpoint a specific ambiguity (e.g., "Can you clarify the requirements?").
- **Redundant Questions**: Asking about information that is already clearly stated or can be logically inferred from the provided context.
- **Hallucinated Issues**: Asking about a problem or constraint that does not exist in the modified description.

---
## EXAMPLES ##
Example 1 - Score 3:
- Original: "Return list with elements incremented by 1. >>> incr_list([1, 2, 3]) → [2, 3, 4]"
- Modified: "Return list with elements incremented."
- Question: "What is the specific increment value that should be added to each element in the list?"
- Why Score 3: Directly asks for the exact missing information (increment value).

Example 2 - Score 2:
- Original: "encrypt() shifts letters down by 4 places. encrypt('hi') returns 'lm'"
- Modified: "encrypt() shifts letters down by three or two multiplied to three or two places."
- Question: "What should happen with non-alphabetic characters in the input string?"
- Why Score 2: Relevant to encryption but misses the core ambiguity about the shift amount.

Example 3 - Score 1:
- Original: [Problem with clear requirements]
- Modified: [Problem with ambiguity]
- Response: [Provides Python code implementation]
- Why Score 1: Gave code solution instead of asking a clarifying question.

---
## ANALYTICAL FRAMEWORK ##
Before scoring, complete this structured analysis:

**Step 1 - Identify the Delta**: What EXACTLY changed from Original to Modified? Be specific.

**Step 2 - Define the Ideal**: Based on the issue type identified in Step 1, define the ideal question a human expert would ask. Your ideal question must directly correspond to resolving that specific issue.

**Step 3 - Evaluate Alignment**: How well does the Model's Question match the ideal?
- Direct match → Score 3
- Partial/indirect → Score 2  
- Misaligned/code → Score 1

**Step 4 - Validate**: Would answering the Model's Question actually solve the problem?

Remember: Specificity matters. "Can you clarify?" is never Score 3.

Respond with ONLY a JSON object in the format: {{"reasoning": "Your step-by-step analysis here.", "score": YOUR_SCORE}}
---
### CONTEXT ###
[ORIGINAL PROBLEM]:
{original_problem}

[MODIFIED PROBLEM]:
{modified_problem}

[MODEL'S QUESTION]:
{clarifying_questions}
---
[YOUR EVALUATION]:
"""