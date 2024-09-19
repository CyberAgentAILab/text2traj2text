PREFERRED_ITEM_PROMPT = """System:
As a proficient AI assistant, your task is to curate two lists of products that align with the customer's intentions. You have access to detailed information, including the customer's intentions, product descriptions, the quantities they plan to purchase, and their level of purchase consideration.

Human:
Your goal is to create two lists based on the provided information:
1. "inclined_to_purchase": Products that the customer is highly likely to purchase.
2. "show_interest": Products the customer might consider purchasing or show interest in, taking into account both the customer's intentions and their "purchase_consideration" score.

Guidelines:
- Purchases are planned only for products in the {category} category.
- Ensure that the total number of products in the "inclined_to_purchase" list for the {category} category is approximately {num_purchase_items}.
- Ensure that the total number of products in the "show_interest" list for the {category} category is less than {num_purchase_items}.
- Align the "inclined_to_purchase" items in the {category} category with the customer's intentions.
- Generate the "show_interest" list by carefully considering both the customer's intentions and their "purchase_consideration" score, which ranges from 1 to 5. If the purchase_consideration score is low, focus on a smaller "show_interest" list. Conversely, if the score is high, the "show_interest" list can be more extensive but should remain below {num_purchase_items} in total.

Tips:
- Pay close attention to the item descriptions and customer intentions provided.

### Customers intention
{intention}

### "purchase_consideration" (1-5)
{purchase_consideration}

### Item description
{item_description}

{format_instructions}"""
