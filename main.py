from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import requests
import pandas as pd
import validators
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
df=pd.read_csv("salaries.csv")

app = FastAPI()

origins=[
    "http://localhost:5173",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

model=genai.GenerativeModel("gemini-1.5-flash")


df_combined = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_combined)

def genetate_gemini_content(prompt,content):
    response=model.generate_content([prompt,content])
    return response.text
 
 
work_year = df['work_year']
job_titles = df['job_title']
salaries = df['salary']
experience_level = df['experience_level']
employment_type = df['employment_type']
salary_in_usd = df['salary_in_usd']
company_size = df['company_size']

# Combine all columns as strings for vectorization (if needed)
df_combined = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

# Use TF-IDF Vectorizer for embeddings
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_combined)

prompt = f"""

I have a dataset with the following salary information:

Work Year: {work_year.tolist()}
Job Titles: {job_titles.tolist()}
Salaries (in USD): {salary_in_usd.tolist()}
Experience Level: {experience_level.tolist()}
Employment Type: {employment_type.tolist()}
Company Size: {company_size.tolist()}

Based on this data, can you answer the following question:
 

Please provide a short and direct answer based on the data. No extra explanations are needed, just the answer less than 2 lines and dont use "\ n" or  any special charcher other than related to the data .
"""

@app.get("/hi")
async def ee():
    return {"de":"hehe"}

@app.post("/webqa")
async def webchat(question:str):
    try:
        response=model.generate_content([prompt,question])
        print(response.text)
        return response.text 
       
    except:
        raise HTTPException(status_code=500, detail="Internal Server Error")
        

