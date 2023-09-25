from gather_data import load_all_posts
from gather_data import Post
from api_keys import openai_api_key
import openai


openai.api_key = openai_api_key


def count_posts(location="998_generated\\"):
    posts = load_all_posts(location=location)
    total_len = 0
    for post in posts[:]:
        total_len += 1#len(post.comment)
    return total_len


def compile_request(post_comment):
    
    query = [{"role": "system", 
         "content": """Hodnotíš príspevky popisujúce stav húb v lese podľa toho, 
         aké dobré podmienky na zber húb momentálne sú. Škála hodnotenia je od 
         1 do 5, kde 1 znamená zlé podmienky a veľmi málo alebo žiadne huby a 5
         znamená výborné podmeinky a veľké množstvo húb. Vraciaš len jeden 
         jediný znak a to číslo medzi 1 a 5."""},
         {"role": "user",
         "content": post_comment}]
    
    return query


def rate_post(post):

    comment = post.comment
    query = compile_request(comment)   
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=query)

response = chat.choices[0].message.content



    
    
    
print()