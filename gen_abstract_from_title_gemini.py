import csv
import json
from pprint import pprint
import google.generativeai as genai
import json 
import os 
from tqdm import tqdm 
import numpy as np
import random
import argparse
import google.generativeai as palm
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import socket
import httpx
   
random.seed(12)
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
os.getenv('GOOGLE_API_KEY')
MYKEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key="")
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b"
)

config = {'candidate_count':1, "max_output_tokens": 2048, "temperature": 0, "top_p": 0.99, "top_k": 32}


file_path_news = './data/MINDlarge_train/new_title.json'

# Reading the TSV file
with open(file_path_news, "r") as file:
        newsInfor = json.load(file)
        print("File loaded successfully!")

prompt = """
Using the title below, generate a short, intriguing abstract that captures its essence and highlights its importance. Keep it under 2–3 sentences.
Title: "{}"
"""

mode = 'train'

file_path_save = f"./data/MINDlarge_{mode}/gen_abs_From_title_small_{mode}_gemini.json"

if os.path.exists(file_path_save):
    with open(file_path_save, "r") as file:
        dict_rs = json.load(file)
        print("File loaded successfully!")
else:
    print(f"The file at {file_path_save} does not exist.")
    dict_rs = {}

import json
# dict_rs = {}
neg_item = []

counter = 0
for idx in tqdm(list(newsInfor.keys())):
    counter += 1
    
    if idx in dict_rs.keys():
        continue
    if mode == 'dev' and idx not in join_id:
        continue
    # # print(id)
    if idx in neg_item:
        continue
    title = newsInfor[idx]
    # print(title)
    # abstract = list_news[i][2]
    prompt_text = prompt.format(title)
    if counter % 60 == 0:
        time.sleep(20)
    print(prompt_text)
    # print(prompt_text)
    responses = model.generate_content(
            prompt_text,
            generation_config=config,
            safety_settings = safety_settings
        )

    if not responses or not hasattr(responses, 'candidates'):
            neg_item.append(idx)
            print(f"No candidates for ID {idx}")
            continue
        
        # Extract text from first candidate
    if responses.candidates:
            candidate = responses.candidates[0]
            if hasattr(candidate, 'text'):
                dict_rs[idx] = candidate.text
            elif hasattr(candidate, 'content') and candidate.content.parts:
                dict_rs[idx] = candidate.content.parts[0].text
            else:
                neg_item.append(idx)
                print(f"No text for ID {idx}")
                continue
    else:
            neg_item.append(idx)
            print(f"Empty candidates for ID {idx}")
            continue

    with open(file_path_save, "w") as file:
        json.dump(dict_rs, file, indent=4)  # indent=4 makes the file more readable
