#pip install transformers faiss-cpu sentence-transformers
#pip install SpeechRecognition
#pip install gTTS
#pip PyAudio.whl
#pip install pygame
import json
from sentence_transformers import SentenceTransformer
import faiss
from gtts import gTTS 
from transformers import pipeline
from io import BytesIO
import speech_recognition as sr  # For voice input
import pygame
# Initialize the TTS engine

sample_rate = 44100  # Sample rate in Hz
# Load the JSON data
with open("askfitness.json", 'r') as f:
    data = json.load(f)

# Extract contexts and responses
contexts = [item['context'] for item in data]
responses = [item['responses'] for item in data]

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the contexts
doc_embeddings = model.encode(contexts, convert_to_tensor=True).cpu().numpy()

# Create a FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Function to retrieve the most relevant context and responses
def retrieve(query, index, model, contexts, responses):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k=1)
    return contexts[indices[0][0]], responses[indices[0][0]]

# Load a text generation model (optional, if you want to refine responses)
generator = pipeline('text-generation', model='gpt2')

# Function to generate a response (optional)
def generate_response(retrieved_context, retrieved_responses, query):
    # You can use the retrieved_context and retrieved_responses to refine the response
    # For example, you could use a language model to generate a response based on the context and responses
    # Here, we're just returning a random response from the retrieved responses
    import random
    return random.choice(retrieved_responses)

# Function to take voice input and convert it to text
def get_voice_input(timeout=10, phrase_time_limit=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for your query...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        
        try:
            # Recognize the speech using Google's speech recognition API
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Error with the Speech Recognition service: {e}")
            return None
        
def tts_and_play(text):
    tts = gTTS(text=text, lang='en')
    # Save the TTS output to a BytesIO stream
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    pygame.mixer.init()
    # Load the audio stream into pygame
    pygame.mixer.music.load(audio_stream,'mp3')
    pygame.mixer.music.play()
    
    # Wait until the playback is finished
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
# Chatbot function
def chatbot():
    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        query = get_voice_input()
        if query is None:
            continue
        if query.lower() == 'exit':
            break
        retrieved_context, retrieved_responses = retrieve(query, index, model, contexts, responses)
        response = generate_response(retrieved_context, retrieved_responses, query)
        print(f"Bot: {response}")
        tts_and_play(response)  # Wait until the playback is finished

# Run the chatbot
if __name__ == "__main__":
    chatbot()
