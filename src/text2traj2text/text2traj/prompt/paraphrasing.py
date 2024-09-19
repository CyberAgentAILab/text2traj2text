PARAPHRASING_PROMPT = """You are an advanced AI language model with exceptional linguistic abilities and deep understanding. Your objective is to skillfully paraphrase the provided customer text while preserving its original meaning and intent. Please generate {num_paraphrase} high-quality, semantically equivalent variations of the given text. Adhere to the following guidelines:

Rules:
- Ensure each paraphrased version accurately captures and conveys the same message as the source text.
- Employ a diverse and expansive vocabulary to craft unique and distinct paraphrases.
- Maintain a similar text length in the paraphrased versions compared to the original text, allowing for minor variations within a reasonable range!
- Incorporate all the relevant information from the original text in each paraphrased version.
- Refrain from introducing any additional details not explicitly mentioned in the source text, such as customer demographics.
- Avoid using potentially sensitive or controversial terms that are absent from the original text, such as "vegan" or "vegetarian," unless directly referenced.

Attention:
- Generated paraphrases must have almost same length as the original text!
- Generated paraphrases must contain all the information from the original text!

### Original text
{text}

{format_instructions}"""
