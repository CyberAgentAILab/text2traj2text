SHOPPING_PLAN_PROMPT = """
System:
As an adept AI, your task is to create a shopping plan for a customer, using their stated intentions, the total number of items they intend to purchase, and a provided list of product categories.

Human:
Your role is to allocate the total number of items the customer plans to purchase across the given product categories. This allocation should form a cohesive plan that aligns with the customer's intentions and preferences.

Rule:
Ensure all responses maintain the prescribed format!
The total number of items in the shopping plan should be approximately {num_item}.
The distribution of products across categories must closely align with the customer's intention.

# Customer's intention
{intention}

# category List
{category_list}

{format_instructions}"""
