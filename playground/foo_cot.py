import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key=openai_api_key,
)


class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


def math_reasoning(question: str) -> MathReasoning:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
            {"role": "user", "content": question},
        ],
        response_format=MathReasoning,
    )

    math_reasoning = response.choices[0].message.parsed

    return math_reasoning


if __name__ == "__main__":
    foo = math_reasoning("how can I solve 8x + 7 = -23")
    print(foo)