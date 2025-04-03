import random
import json
import torch
from fuzzywuzzy import fuzz
from metaphone import doublemetaphone
from better_profanity import profanity

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SIES"

# List of known abusive words (expand this list)
abusive_words = ["bhenchod", "madarchod", "chutiya", "harami", "gaand", "kutti", "suar", "chakka"]

# ✅ List of safe words (to prevent false positives)
safe_words = [
    "hello", "hi", "good morning", "good afternoon", "good evening", 
    "good night", "how are you", "thank you", "bye", "goodbye", "see you"
]

def contains_abusive_words(msg):
    """Detects abusive words using exact match, fuzzy matching, and phonetic similarity"""
    msg_lower = msg.lower().strip()

    # ✅ Step 1: Ignore safe words
    if msg_lower in safe_words:
        return False  

    words = msg_lower.split()

    # ✅ Step 2: Check for fuzzy matches (only for words longer than 3 letters)
    for word in words:
        if len(word) < 4:  # Ignore short words to prevent false matches
            continue
        for abuse in abusive_words:
            if fuzz.ratio(word, abuse) > 90:  # Increased threshold to 90%
                return True

    # ✅ Step 3: Check for phonetic similarity (only for words longer than 3 letters)
    for word in words:
        if len(word) < 4:  # Ignore short words
            continue
        word_sound = doublemetaphone(word)[0]  # Get phonetic encoding
        for abuse in abusive_words:
            abuse_sound = doublemetaphone(abuse)[0]
            if word_sound == abuse_sound:  # Compare phonetic encoding
                return True

    # ✅ Step 4: Check for direct match using `better-profanity` (last step to avoid false positives)
    if profanity.contains_profanity(msg_lower):
        return True

    return False  # If no abuse detected

def get_response(msg):
    """Process user message and respond accordingly"""
    # 🚨 First, check if the message is abusive
    if contains_abusive_words(msg):
        return "⚠️ Warning: Please maintain a respectful conversation!"

    # Normal chatbot processing
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."

# Chat loop
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        response = get_response(sentence)
        print(f"{bot_name}: {response}")
