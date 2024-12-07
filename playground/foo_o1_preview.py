#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/03 22:09
@Author  : fei.wang@iu.org
@File    : foo_o1_preview.py
@Modified by : USER on 2024/11/03
@Comment : test openai o1 preview model
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=openai_api_key)


def test_openai_o1_preview():
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user",
                "content": "Write a bash script that takes a matrix represented as a string with format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.",
            }
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    test_openai_o1_preview()
