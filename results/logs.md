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
| **Training time** | ~1hour |

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
| **Training time** | ~3.5 hour |

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
Once upon a time when he joined the London club The Str aw ber ry , and
was joined by his former club manager . In the mid - 2000s , he was manager
of Peter O ' Bri en , manager of the club ' s Mid lands . In the summer of
2005 , O ' Bri en was an assistant manager and manager of the club , as
well as manager of the club , the club ' s owner , Rob bie H oy man . In
May 2006 O ' Bri en was appointed manager of Manchester City , signing a
one - year contract with a £ 2 . 0 million contract from the club . After
his retirement , O ' Bri en was the manager of East R iding .
---
>>> The most important thing in life is
The most important thing in life is a description of the work on the
philosopher Paul a L ig man . The story was edited by Paul ine L ig man .
In his book , Paul ine L ig man wrote in the book : The Way of L aughter :
The Mind of the King James and the Lost Man . The first film was published
in 1973 , starring N ancy N ort on .
---
>>> I do not know
I do not know what how to understand the " b - sides a good or a good to
be for us ". The first episode of O y ster , a television series featuring
a young artist , is shot in the box office . The series features a story
by the artist , who thin ks of his physical inter ception .
---
>>> Language is
Language is the third person ( a person who has been a member of the
family ) in terms of the number of individuals whose name is derived .
This is the third person in the family , so does not distinguish between
the remaining four members in the family .
```


## v3 — Diverse Data

| | |
|---|---|
| **config** | same as v1 (D_MODEL=256, N_LAYERS=12, N_HEADS=4, FFN_DIM=768) |
| **params** | 14,326,016 |
| **dataset** | Wikipedia + TinyStories + FineWeb |
| **tokens** | 600M+ (train) |
| **optimizer** | AdamW, β=(0.9,0.95), λ=0.1 |
| **schedule** | linear warmup 2000 steps → cosine decay |
| **peak LR** | 3e-3 |
| **batch** | 16 × 1024 = 16,384 tok/step |
| **steps** | ~24,000 |
| **dtype** | bfloat16 |
| **Training time** | ~4 hour |

**Reports:**
- coherent multi-sentence narratives with dialogue and story arcs
- significantly less repetition — full paragraphs without looping
- proper character names, consistent references across sentences
- shows understanding of basic story structure (setup → conflict → resolution)
- still hallucinates facts but the language is natural and fluent

**Outputs:**
```
>>> Once upon a time
Once upon a time , there was a little girl named L ily . She lov ed to
play outside and run around outside . One day , she saw a boy playing with
his friend Johnny . She ran up to him and said , " H i , what ' s your
name ?" Johnny replied , " I ' m L ily . W ould you like to play with me
and my to ys ?" L ily was so ex cited and said , " Y es , ple ase !" They
played together for a while until they got t ired . As they were playing ,
they saw a big and sc ary dog . L ily said , " Let ' s play a game ! It
will make l oud faces and sc ary things happen ." Johnny replied , " No ,
this is a
---
>>> The most important thing in life is
The most important thing in life is to measure the truth of the heart ,
and to make sure the truth is in the sense of the world . One day , when
you measure the truth , you will be pro ud of your self . And if you
measure . A lot of things with the truth , and if you learn something ,
you will have to change it and become more important in time . A very
special memory of the truth is that he has done a very important thing .
Every one who tal ked about it and how he has measured . That is the truth
of the truth . You must know that these things are important and that we
don ' t need to make mist akes . If you want to prove this to me , I will
make you a very important memory . In practice
---
>>> I do not know
I do not know what he was saying . When the next day came , he went back
to the p ond and went to the water . He started the water and slowly
turned the water around . Soon he saw that the water was rising . It had
made the water rise as fast as he could . He was so sur pr ised ! But then
he heard a voice that said , " I ' m sor ry I went to the water and it
broke the water ! Now the water is rising and the water is rising to the
water ." The fox was very happy and had to pay attention to the water .
But he knew even though it was too hard his to ugh ness he was so sor ry .
---
>>> Language is
Language is a language in the South America . It has a total area ( 12 9 ,
9 21 inhabitants ), an area ( 27 7 , 6 19 inhabitants ), and a total area
( 5 , 3 21 inhabitants ), in the 2016 population density . It is the home
to the International E cu ador ian Forest , a mountain range . Geography
The climate of the city is . History The region was established in 185 3
when it was created by the French colonial Empire . It was named the
" Grand White Kingdom ".
---
>>> A person who always learns
A person who always lear ns to love each other . Once upon a time , there
was a little girl named L ily . She lov ed to play with her dol ls and
read books . One day , L ily ' s mom my asked her to help dress up her
hair . L ily got to work and dress ed herself right away . She was going
to dress up right away . As she was dress ing , her friend , Tim my , came
to see what was wrong . Tim my said , " L ily , why do you dress up ?"
L ily said , " I wear it because it makes me sad ." Tim my said , " Don '
t wor ry , L ily . You are special to me ." But L ily didn ' t want to
dress up right away
```
