import os
from dotenv import load_dotenv #type:ignore
from google import genai
from google.genai.types import GenerateContentConfig #type:ignore

load_dotenv()

def generate_cognitive_report(bpm, sleep, work, noise, context, stress_label):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    sys_instruction = (
        "You are an expert clinical psychologist and cognitive scientist specializing in occupational stress. "
        "Your task is to analyze objective physiological data alongside subjective user context. "
        "Provide a structured, highly accurate explainable AI diagnostic report. "
        "Validate the machine learning model's prediction by mapping how sleep deficits, physiological metrics (BPM), "
        "and environmental stressors compound cognitive load. Keep the analysis clinically sound, supportive, and concise."
    )
    
    prompt = (
        f"Data Analysis Request:\n"
        f"- Heart Rate: {bpm} BPM\n"
        f"- Sleep Duration: {sleep} hours\n"
        f"- Continuous Work: {work} hours\n"
        f"- Noise Level (1-3): {noise}\n"
        f"- User Context: {context}\n"
        f"- ML Model Classification: {stress_label}"
    )
    
    config = GenerateContentConfig(
        system_instruction=sys_instruction,
        temperature=0.2
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    return response.text