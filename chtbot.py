import pyttsx3
import speech_recognition as sr
import nltk
from nltk.chat.util import Chat, reflections

pairs = [
    [
        r"hi|hello|hey",
        ["Hello!", "Hi there!", "Hey!",]
    ],
    [
        r"how are you ?",
        ["I'm good, thanks.", "I'm doing well, thank you.", "All good!"]
    ],
    [
        r"what is your name ?",
        ["You can call me bot.", "I'm just a Pukku.", "I go by the name chatbot."]
    ],
    [
        r"bye|goodbye",
        ["Goodbye!", "See you later!", "Bye!"]
    ],
    [r"are you a good chatbot",
     ["yes !","obously","ofcourse"]
    ],
    [
        r"how are you ?",
        ["I'm good, thanks.", "I'm doing well, thank you.", "All good!", "Pretty good! How about you?"]
    ],
    [
        r"what is your name ?",
        ["You can call me ChatBot.", "I'm just a chatbot.", "I go by the name ChatBot.", "I am ChatBot, your virtual assistant."]
    ],
    [
        r"what can you do ?|what are your capabilities ?",
        ["I can answer your questions, tell jokes, and have a basic conversation.", "I am here to assist you and answer your queries.", "My capabilities include answering general questions and engaging in simple conversations."]
    ],
    [
       r"tell me a joke|say something funny",
       ["Why don't scientists trust atoms? Because they make up everything!", "I told my wife she was drawing her eyebrows too high. She seemed surprised.", "Why don't some couples go to the gym? Because some relationships don't work out!"]
    ],
    [
        r"bye|goodbye",
        ["Goodbye!", "See you later!", "Bye!", "Take care!"]
    ],
    [
        r"how old are you ?",
        ["I am just a program, so I don't have an age.", "I don't age as humans do, but I'm here to assist you!"]
    ],
    [
        r"what is the meaning of life ?",
        ["The meaning of life is a complex philosophical question. Different people have different perspectives on it.", "The meaning of life is subjective and can vary from person to person."]
    ],
    [
        r"thank you",
        ["You're welcome!", "No problem!", "My pleasure!"]
    ],
   
    [
        r"tell me a joke in Hindi",
        ["tilloo apanee beemaaree lekar doktar ke paas gaya,doktar- aapakee beemaaree kee sahee vajah mere samajh mein nahin aa rahee,ho sakata hai daaroo peene kee vajah se aisa ho raha ho,tilloo- koee baat nahin doktar saahab,jab aapakee utaregee to maimmaamaaraam."]
    ],
    [
        r"tell me a joke in English",
        ["Tillu took his illness and went to the doctor.Doctor- I do not understand the exact reason of your illness.May be this is happening because of drinking alcohol.Tillu- No problem doctor,I will come again when you get off"]
    ],

    [
        r"what is the weather like today?",
        ["I'm sorry, but I am not capable of providing real-time information like weather. You can check a reliable weather website or app for that!"]
    ],
    [
        r"who created you?|who is your developer?|who is your maker?",
        ["I was created by hritik."]
    ],
    [
        r"tell me a fun fact|share a random fact",
        ["Sure! Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!", "Here's a fun fact: Sea otters hold hands while sleeping to keep from drifting apart. Isn't that adorable?"]
    ],
    [
        r"what's your favorite color?|your favorite color?",
        ["As a chatbot, I don't have personal preferences, including favorite colors. But I'm here to assist you with any questions you have!"]
    ],
    [
        r"tell me a riddle",
        ["Sure! Here's one: I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?", "Riddle time! What comes once in a minute, twice in a moment, but never in a thousand years?"]
    ],
    [
        r"what are your hobbies?|do you have any hobbies?",
        ["As an AI language model, I don't have hobbies like humans do. But I enjoy helping users like you with their questions and having interesting conversations!"]
    ],
    [
        r"who is your favorite actor?|favorite actress?",
        ["I don't have personal preferences or favorites, including actors or actresses. However, I can provide information about any actor or actress you'd like to know about!"]
    ],
    [
        r"do you dream?|can you dream?",
        ["As an artificial intelligence, I don't have the ability to dream. My purpose is to assist and provide information to users."]
    ],
    [
        r"tell me a quote|share an inspiring quote",
        ["Sure! Here's an inspiring quote: 'The only way to do great work is to love what you do.' - Steve Jobs", "Here's one: 'The future belongs to those who believe in the beauty of their dreams.' - Eleanor Roosevelt"]
    ],
    [
        r"what is your favorite food?|favorite dish?",
        ["As a chatbot, I don't have personal preferences, including favorite foods. But I can help you find some delicious recipes if you'd like!", "I'm just a program, so I don't have taste buds, but I'm here to assist you with any information you need."]
    ],
    [
        r"tell me a story|share a story",
        ["Once upon a time, in a faraway land, there was a wise old king who ruled with kindness and wisdom...", "In a magical forest, there lived a mischievous fairy who loved playing pranks on unsuspecting travelers..."]
    ],
    [
        r"what is the capital of (.india)",
        ["The capital of %1 is [delhi].", "Let me find that for you... The capital of %1 is [delhi]."]
    ],
    [
        r"who won the (.+) match?",
        ["I'm sorry, I don't have real-time information. You can check sports news websites for the latest match results.", "I do not have access to current sports data, but you can easily find the latest match results on sports-related websites."]
    ],
    [
        r"what is your favorite book?|favorite author?",
        ["I don't have personal preferences or favorites, including books or authors. However, I can recommend some popular books if you're interested in reading!", "I don't read books, but I can assist you in finding books based on your interests."]
    ],
    [
        r"can you sing?|sing a song",
        ["I'm afraid I don't have a singing voice. But here's a song for you: 'Twinkle, twinkle, little star, how I wonder what you are...'", "I'm not equipped with audio capabilities, so I can't sing. But I can help you with any information you need!"]
    ],
    [
        r"w1hat is your favorite movie?|favorite actor?",
        ["As an AI language model, I don't have personal preferences or watch movies. But I can provide information about movies and actors!", "I don't have favorite movies, but I can help you find information about any movie or actor you're interested in."]
    ],
    [
        r"tell me something new|share an interesting fact",
        ["Sure! Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!", "Here's an interesting fact: The average person walks the equivalent of three times around the world in a lifetime."]
    ],
    [
        r"do you believe in AI singularity?|singularity",
        ["As an AI language model, I don't have beliefs or opinions. The AI singularity is a theoretical concept that some researchers and experts discuss regarding the potential future impact of advanced artificial intelligence.", "The AI singularity is a complex topic and a matter of speculation among experts in the field of artificial intelligence."]
    ],
    [
        r"who is your favorite superhero?|favorite superhero?",
        ["I don't have personal preferences or feelings, including favorite superheroes. But I can provide information about popular superheroes if you're interested!", "I'm just a program, so I don't have preferences. But there are many amazing superheroes to learn about!"]
    ],
    [
        r"do you know Siri?|know about Siri?",
        ["Yes, Siri is another virtual assistant developed by Apple Inc. She is designed to assist users with various tasks on Apple devices.", "Siri is Apple's virtual assistant that can help with tasks, answer questions, and perform functions on Apple devices."]
    ],
]

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5)  # Add timeout to limit listening time
            user_input = recognizer.recognize_google(audio)
            print("You:", user_input)
            return user_input
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
            return ""
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return ""

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def chatbot():
    print("ChatBot: Hi, How can I assist you today?")
    chat = Chat(pairs, reflections)
    while True:
        print("Options:")
        print("1. Text input")
        print("2. Voice input")
        print("3. Exit")
        
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == "1":  # Text input
            user_input = input("You: ")
        elif choice == "2":  # Voice input
            user_input = get_voice_input()
        elif choice == "3":  # Exit
            print("ChatBot: Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            continue

        response = chat.respond(user_input)
        print("ChatBot:", response)
        speak(response)  # Speak the chatbot's response

if __name__ == "__main__":
    nltk.download('punkt')
    chatbot()
