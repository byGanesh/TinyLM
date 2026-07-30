# Experiment Logs

## v1 — Baseline

| | |
|---|---|
| **config** | `src/config.py` (D_MODEL=256, N_LAYERS=12, N_HEADS=4, FFN_DIM=768) |
| **params** | 14,326,016 |
| **dataset** | Simple Wikipedia (20231101.simple) |
| **tokens** | 62,914,560 (train) |
| **optimizer** | AdamW, β=(0.9,0.95), λ=0.1 |
| **schedule** | linear warmup 2000 steps → cosine decay |
| **peak LR** | 3e-3 |
| **batch** | 16 × 1024 = 16,384 tok/step |
| **steps** | 3840 |
| **dtype** | bfloat16 |
| **val loss** | 3.68 |
| **val ppl** | 39.6 |

**Loss:**  

| step | train loss | train ppl | val loss | val ppl |
|-----:|----------:|----------:|--------:|--------:|
| 500 | 4.99 | 146.3 | 5.63 | 279.5 |
| 1000 | 4.16 | 64.4 | 4.91 | 136.0 |
| 2000 | 3.60 | 36.6 | 4.24 | 69.5 |
| 3000 | 3.18 | 24.0 | 3.78 | 43.9 |
| 3840 | 3.04 | 21.0 | 3.68 | 39.6 |

**Reports:**
- repetitive loops, copies words/phrases within 2-3 sentences
- hallucinated entities are generic or contradictory (e.g. Nixon timeline)
- coherence degrades quickly — output becomes circular
- limited vocabulary, simple sentence structures

**Outputs:** 
```
>>> Once upon a time
Once upon a time , they were to re tain their new s essions . After their death in 2011 , the first time since the death of a woman , Dr . David , was born on the wedding of the first woman , and her husband ' s brother , J . S . Wilson . The couple were married to the United States President Richard Nixon . The couple divorced in 2012 , and their three children were : D . D . R . Wilson , Jr . ( d . 2018 ) George Washington , U . K .
---
>>> The most important thing in life is
The most important thing in life is the first to believe . It is the main reason why people think it is a way of thinking about it for what people think of . This is called the " God " which means " the truth ." The idea of life is the " God ". The name of life comes from a lot of different sources . The word is sometimes used in the books by the authors and authors . For example , the New Testament is an important part of the story . It was first written in the New Testament in 150 6
---
>>> I do not know
I do not know about the problems , and the others are a common problem . The average number of people living in the world is about 6 . 6 % of the population . The number of people living in the world is about 2 . 5 % ( 1 . 9 % ). The number of people living in the world is above the rate that the disease or the number of people living in the world is below the rate . The number of people living in the world is about 2 . 3 % of people living in the world
---
>>> Language is
Language is the word for a person . L ingu istics comes from the Greek language as a word for a person . People like L ingu istics are different from people to people or women . People who like to live in a place should make people sick or sick or sick or other parts of they will get to people from the same way . People who who like to live there are called other people , but people sometimes have other problems if they are sick or sick or have their own mind . People who do not like
```

## v2 — More Tokens

| | |
|---|---|
| **config** | same as v1 (D_MODEL=256, N_LAYERS=12, N_HEADS=4, FFN_DIM=768) |
| **params** | 14,326,016 |
| **dataset** | Simple Wikipedia (20231101.simple) |
| **tokens** | 163,840,000 (train) |
| **optimizer** | AdamW, β=(0.9,0.95), λ=0.1 |
| **schedule** | linear warmup 2000 steps → cosine decay |
| **peak LR** | 3e-3 |
| **batch** | 16 × 1024 = 16,384 tok/step |
| **steps** | 10,000 (resumed from 3840) |
| **dtype** | bfloat16 |
| **val loss** | 3.74 |
| **val ppl** | 41.9 |

**Loss:**

| step | train loss | train ppl | val loss | val ppl |
|-----:|----------:|----------:|--------:|--------:|
| 4000 | 5.42 | 225.0 | 5.23 | 187.5 |
| 4500 | 4.43 | 84.3 | 4.40 | 81.9 |
| 5000 | 4.25 | 69.8 | 4.25 | 69.9 |
| 5500 | 4.11 | 61.2 | 4.11 | 60.8 |
| 6000 | 4.02 | 55.5 | 4.04 | 56.8 |
| 6500 | 3.95 | 51.8 | 3.97 | 52.8 |
| 7000 | 3.90 | 49.6 | 3.92 | 50.2 |
| 7500 | 3.86 | 47.6 | 3.88 | 48.5 |
| 8000 | 3.80 | 44.6 | 3.84 | 46.7 |
| 8500 | 3.77 | 43.4 | 3.80 | 44.8 |
| 9000 | 3.73 | 41.5 | 3.79 | 44.3 |
| 9500 | 3.68 | 39.8 | 3.74 | 41.9 |

**Reports:**
- noticeably better coherence; maintains topic for longer spans
- more varied vocabulary and specific named entities (clubs, people, titles)
- still hallucinates but with plausible structure (e.g. football manager narrative)
- less immediate repetition, smoother transitions between phrases

**Outputs:**  
```
>>> Once upon a time
Once upon a time , they were to re tain their new s essions . After their death in 2011 , the first time since the death of a woman , Dr . David , was born on the wedding of the first woman , and her husband ' s brother , J . S . Wilson . The couple were married to the United States President Richard Nixon . The couple divorced in 2012 , and their three children were : D . D . R . Wilson , Jr . ( d . 2018 ) George Washington , U . K .
---
>>> The most important thing in life is
The most important thing in life is the first to believe . It is the main reason why people think it is a way of thinking about it for what people think of . This is called the " God " which means " the truth ." The idea of life is the " God ". The name of life comes from a lot of different sources . The word is sometimes used in the books by the authors and authors . For example , the New Testament is an important part of the story . It was first written in the New Testament in 150 6
---
>>> I do not know
I do not know about the problems , and the others are a common problem . The average number of people living in the world is about 6 . 6 % of the population . The number of people living in the world is about 2 . 5 % ( 1 . 9 % ). The number of people living in the world is above the rate that the disease or the number of people living in the world is below the rate . The number of people living in the world is about 2 . 3 % of people living in the world
---
>>> Language is
Language is the word for a person . L ingu istics comes from the Greek language as a word for a person . People like L ingu istics are different from people to people or women . People who like to live in a place should make people sick or sick or sick or other parts of they will get to people from the same way . People who who like to live there are called other people , but people sometimes have other problems if they are sick or sick or have their own mind . People who do not like
```
