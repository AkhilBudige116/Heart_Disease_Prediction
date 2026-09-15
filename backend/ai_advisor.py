"""Generate cautious, educational health guidance after a prediction."""

import os

from groq import Groq

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_ai_precautions(
    result_text: str,
    age,
    cholesterol,
    max_heart_rate,
    blood_pressure,
    old_peak,
) -> str:
    """Return educational precautions for either prediction outcome."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from the .env file.")

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)
    prompt = f"""A heart-disease prediction app returned: {result_text}.

Patient values:
- Age: {age}
- Cholesterol: {cholesterol}
- Maximum heart rate: {max_heart_rate}
- Blood pressure: {blood_pressure}
- Old peak: {old_peak}

Give 4-6 short, practical precautionary suggestions based on these values and
the prediction. Give useful preventive guidance even when the result is \"No
Heart Disease Detected\". Do not diagnose, prescribe medication, or claim the
numbers establish a condition. Include a brief note to discuss individual risk
with a licensed clinician, and advise urgent/emergency care for chest pain,
trouble breathing, fainting, or other severe sudden symptoms."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You provide cautious general health education. Do not replace "
                    "a clinician, provide a diagnosis, or prescribe treatment."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=350,
    )
    return response.choices[0].message.content
