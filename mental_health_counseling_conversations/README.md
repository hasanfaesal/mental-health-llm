---
license: other
license_name: rail-d
license_files: LICENSE-RAIL-D.txt
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- medical
size_categories:
- 1K<n<10K
pretty_name: Amod - Mental Health Counseling Conversations
---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64148f132dfc95f58895052f/XW8KQD9wpUn_ihACynGyZ.png)

# Amod/mental_health_counseling_conversations

This dataset is a compilation of high-quality, real one-on-one mental health counseling conversations between individuals and licensed professionals. Each exchange is structured as a clear question–answer pair, making it directly suitable for fine-tuning or instruction-tuning language models that need to handle sensitive, empathetic, and contextually aware dialogue.

Since its public release in 2023, it has been downloaded over 100,000 times (As of Nov 2025); hitting 10,000+ downloads in Nov 2025 alone. The data is provided in a clean format, allowing for straightforward integration into training pipelines with minimal preprocessing.

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Licence and Commercial-Use Terms](#licence-and-commercial-use-terms)

## Dataset Description

- **Point of Contact:** amodsahabandu@icloud.com

### Dataset Summary
This dataset is a collection of real counselling question-and-answer pairs taken from two public mental-health platforms.  
It is intended for training and evaluating language models that provide safer, context-aware mental-health responses.

### Supported Tasks
Text generation and question-answering with an advice-giving focus.

### Languages
English (en)

## Dataset Structure

### Data Instances
Each instance contains:
- **Context** – the user’s question  
- **Response** – the psychologist’s answer  

### Data Fields
- `Context` *:string*  
- `Response` *:string*  

### Data Splits
No predefined splits. Users may create their own.

## Dataset Creation

### Curation Rationale
Created to advance AI systems that deliver compassionate, evidence-based mental-health guidance.  
All data were anonymised and retained verbatim to preserve conversational integrity.

### Source Data
Collected directly from two publicly accessible counselling websites; no private or paid sources were used.

### Personal and Sensitive Information
Content is sensitive by nature. All personally identifiable information has been removed during curation.

## Licence and Commercial-Use Terms

This dataset is released under **RAIL-D**.

**Free non-commercial research use**  
Academic, scientific, educational and other non-profit uses are royalty-free. Users must comply with the ethical restrictions in `LICENSE-RAIL-D.txt`.

**Commercial use — mandatory donation**  
Any commercial use requires a donation of **USD 100 or more** to the CCC Foundation mental-health helpline.  
Donation page: <https://1333.lk/donations>  
Email proof of donation to **amodsahabandu@icloud.com** within thirty (30) days before or after first commercial deployment.  
Commercial rights terminate automatically if proof is not provided.

**No content modification**  
Filtering or subsetting is allowed, but individual question-and-answer pairs must not be rewritten, deleted, or altered.

**Redistribution**  
Permitted only in the original, unmodified form and with this licence attached.

The full legal text is provided in `LICENSE-RAIL-D.txt` within this repository.