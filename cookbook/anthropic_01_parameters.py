#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/07 18:39
@Author  : fei.wang@iu.org
@File    : anthropic_01_parameters.py
@Modified by : USER on 2024/12/07
@Comment : play with anthropic parameters
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()


llm_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("IU_LLM_ENDPOINT"),
    api_key=os.getenv("IU_API_KEY"),
    api_version="2024-10-21",
)

async def test_max_tokens():
    sys_prompt = """
    You are a helpful assistant that will help the user for different tasks.
    """
    response =  await llm_client.chat.completions.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=800, 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "write me a long paragraph about the importance of the sun in our lives"},
        ],
        temperature=0.95,
        # stop=['poem']
    )
    print(response.choices[0].message.content)
    # api limites, performance, and response quality can be affected by this parameter
    # if it is too low, the model may not have enough tokens to generate a response
    # if it is too high, the model may generate a response that is too long or off-topic



async def main():
    await test_max_tokens()


if __name__ == "__main__":
    print("Hello, anthropic parameters!")
    asyncio.run(main())
