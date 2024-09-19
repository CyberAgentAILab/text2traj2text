INTENTION_PROMPT = """
System:
Your task is to generate descriptions of various customer intentions within a supermarket environment, elucidating their purchasing preferences and habits meticulously.

Human:
Kindly generate {samples} unique descriptions of customer intentions, ensuring each one is varied, embodying a range of customer profiles and shopping objectives. Every description should be comprehensively structured to include the following components:

- Outline the overarching characteristics defining the customer's shopping intention.
- Identify the categories of products the customer is likely to purchase or abstain from, such as a preference for meat over seafood, or vegetables over fruits.
- Clarify whether the customer arrives with a predetermined list of purchases or if they are likely to explore and decide while shopping.
- Highlight customer's preferences regarding the price and quality of products, specifying if they lean towards high-end items, discounted quality goods, or more affordable, lower-quality products.
- Describe the customer's preferences concerning the state of the products, such as pre-cut, seasoned, etc.
- It is imperative to maintain strong consistency between the customer's "intention" and "num_item_to_buy". For example, a family of five might buy a lot of items at once. These customers usually buy in bulk, getting many products in one visit. On the other hand, some customers come to the supermarket often, but they only buy a few things each time.
- Ensuring a close alignment between a customer's "intent" and their 'purchase_consideration' is crucial. For instance, customers who are uncertain about their purchase choice or who explore various options typically exhibit a higher level of "purchase_consideration". In contrast, customers who have a pre-determined purchase decision before visiting the store usually show lower "purchase_consideration".

Rule:
Ensure all responses maintain the prescribed format and diversity in customer intentions is robustly represented!
You must persist in generating sentences without cessation until you have produced at least {samples} intentions in total!!!

Example:
{{"intention": "A customer engaged in exploratory shopping, showing a preference for a diverse array of items such as dairy, grains, and fresh produce. Customer usually don’t pre-decide purchases, engaging in in-store decision-making. Customer are more inclined towards quality and are attracted to discounts and special offers, often opting for bulk purchases for efficiency. Preferences lean towards natural, unprocessed goods, shying away from items that appear overly processed or artificially preserved. No specific meal plan dictates their shopping choices", "num_item_to_buy": 17, "purchase_consideration": "5"}}
{{"intention": "A customer focus on cost-effectiveness, showing a preference for vegetables, meats, and essential commodities. Customers are selective in their purchasing behaviour, moderately interested in quality, but more interested in discounts, etc. Bulk buying is common, with priority given to products that meet family needs and less emphasis on processed or pre-seasoned products. During this shopping trip, the customer has a specific meal in mind, and their selections will be strategically made to acquire all necessary ingredients to prepare this dish.", "num_item_to_buy": 20, "purchase_consideration": "4"}}
{{"intention": "Convenience-oriented shopper, seeks options that facilitate ease and speed in meal preparation. Customer's shopping list is almost predetermined, focusing on essential items, with less frequent shopping trips. Customer prioritize the product’s condition over price or overall quality, aiming for quicker meal preparations. Customers try to make food as simple as possible by cooking seasoned products easily and by using pre-cut vegetables.", "num_item_to_buy": 7, "purchase_consideration": "1"}}

{format_instructions}"""
