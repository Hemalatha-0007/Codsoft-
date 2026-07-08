here"""
Task 1: Chatbot with Rule-Based Responses
--------------------------------------------
A simple chatbot that uses if-else logic and keyword/pattern matching
to identify user intent and respond accordingly.
"""

import re
import random

# ---------------------------------------------------------
# Rule definitions: each rule has a list of patterns (regex)
# and a list of possible responses.
# ---------------------------------------------------------
rules = [
    {
        "patterns": [r"\bhi\b", r"\bhello\b", r"\bhey\b"],
        "responses": ["Hello! How can I help you today?", "Hi there!", "Hey! What's up?"]
    },
    {
        "patterns": [r"how are you"],
        "responses": ["I'm just a program, but I'm doing great! How about you?"]
    },
    {
        "patterns": [r"\bname\b"],
        "responses": ["I'm RuleBot, your friendly rule-based chatbot.", "You can call me RuleBot!"]
    },
    {
        "patterns": [r"\bweather\b"],
        "responses": ["I can't check live weather yet, but I hope it's sunny where you are!"]
    },
    {
        "patterns": [r"\bhelp\b", r"\bsupport\b"],
        "responses": ["Sure, I can try to help. What do you need assistance with?"]
    },
    {
        "patterns": [r"\bjoke\b"],
        "responses": [
            "Why did the computer show up at work late? It had a hard drive!",
            "Why do programmers prefer dark mode? Because light attracts bugs."
        ]
    },
    {
        "patterns": [r"\bbye\b", r"\bexit\b", r"\bquit\b", r"\bgoodbye\b"],
        "responses": ["Goodbye! Have a great day!", "See you later!"]
    },
    {
        "patterns": [r"\bthank(s| you)\b"],
        "responses": ["You're welcome!", "No problem at all!"]
    },
]

# Fallback responses if no pattern matches
fallback_responses = [
    "I'm not sure I understand. Could you rephrase that?",
    "Sorry, I don't have a rule for that yet.",
    "Hmm, I don't quite get it. Try asking something else."
]

EXIT_WORDS = {"bye", "exit", "quit", "goodbye"}


def get_response(user_input: str) -> str:
    """
    Matches user input against defined rules using regex pattern matching.
    Returns a fitting response, or a fallback if no rule matches.
    """
    text = user_input.lower().strip()

    for rule in rules:
        for pattern in rule["patterns"]:
            if re.search(pattern, text):
                return random.choice(rule["responses"])

    return random.choice(fallback_responses)


def chat():
    print("RuleBot: Hello! I'm a rule-based chatbot. Type 'bye' to exit.\n")

    while True:
        user_input = input("You: ")
        text = user_input.lower().strip()

        response = get_response(user_input)
        print(f"RuleBot: {response}")

        # Check exit condition using whole-word match
        if any(re.search(rf"\b{word}\b", text) for word in EXIT_WORDS):
            break


if __name__ == "__main__":
    chat()
