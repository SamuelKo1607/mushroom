from gather_data import load_all_posts
from gather_data import load_list
from gather_data import save_list
from gather_data import Post
from api_keys import openai_api_key
import sys
import openai
import re
import pickle
import os
import numpy as np


openai.api_key = openai_api_key


def count_posts_words(location="998_generated\\"):
    """
    A simple function to load all the posts 
    and to count the number of words in those.

    Parameters
    ----------
    location : str, optional
        The relative path to the data folder.

    Returns
    -------
    total_len : int
        The total number of words in al lthe posts.

    """
    
    posts = load_all_posts(location=location)
    total_len = 0
    for post in posts[:]:
        total_len += len(post.comment.split())
    return total_len


def compile_request(post_comment,legacy = False):
    """
    A function to put a post comment in a query fromat for GPT API chat.

    Parameters
    ----------
    post_comment : str
        The comment to rate.
    legacy : bool, optional
        Whether to use legacy, more explicit but more token-expensive 
        task description. The default is False.

    Returns
    -------
    query : list of dict.
        The querry that GPT API can process.

    """
    if legacy:
        query = [{"role": "system", 
              "content": """Hodnotíš príspevok popisujúci stav húb v lese podľa toho, 
              aké dobré podmienky na zber húb prípsevok popisuje. Škála hodnotenia je od 
              1 do 5, kde 1 znamená zlé podmienky a veľmi málo alebo žiadne huby a 5
              znamená výborné podmeinky a veľké množstvo húb. Chcem veľmi krátku odpoveď, 
              a to len jeden jediný numerický znak, teda číslo medzi 1 a 5."""},
              {"role": "user",
              "content": post_comment}]
    else:         
        query = [{"role": "system", 
             "content": """Hodnotíš komentár podľa toho, aké dobré hubárske podmienky popisuje. Hodnotíš od 1 do 5, kde 1 znamená veľmi málo a 5 veľmi veľa húb. Odpovedaj číslom medzi 1 a 5."""},
             {"role": "user",
             "content": post_comment}]
        
    return query


def rate_post(post):
    """
    A function to assign a rating to a post. The raiting is supposed to 
    evaluate the mushroom picking conditions. If the post is marked as 
    negative by the author, the raiting is 1.

    Parameters
    ----------
    post : Post type object
        A post.

    Raises
    ------
    Exception
        If raiting can't be assigned.

    Returns
    -------
    rate
        The raiting 1-5 ascending.

    """
    if not post.is_positive:
        return 0
    else:
        comment = post.comment
        query = compile_request(comment)   
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=query)
        response = chat.choices[0].message.content
    
        rates = re.findall(r'\d+', response)
        if len(rates)>1:
            while True:
                #will loop infinitely if there are several numbers and none of them fits
                rate = int(np.random.choice(rates))
                if 0<rate<6:
                    return rate
        elif len(rates)==0:
            raise Exception("no rating")
        else:        
            return int(rates[0])


def rate_selection(start=10000,
                   end=10010,
                   location="998_generated\\"):
    """
    Not for production, but loads all the posts and rates and prints 
    the rating of a selection of all the impacts. 

    Parameters
    ----------
    start : int, optional
        The index of the first analyzed (inclusive) post. 
        The default is 10000.

    ed : int, optional
        The index of the last analyzed (exclusive) post. 
        The default is 10010.

    location : str, optional
        The filder with all the posts that may be analyzed. 
    The default is "998_generated\\".

    Returns
    -------
    None
    """
    posts = load_all_posts(location=location)[start:end]
    for post in posts:
        response = rate_post(post)
        post.print_post()
        print("---------------------------------")
        print(response)
        print("---------------------------------")
        print("---------------------------------")
        
             
def add_one_rate(location="998_generated\\"):
    """
    Adds one rating to a random post and saves.
    
    Parameters
    ----------
    location : str, optional
        The filder with all the posts that may be analyzed. 
        The default is "998_generated\\".
    
    Returns
    -------
    None
    """
    filelist = os.listdir(location)
    order = np.random.choice(filelist,len(filelist),replace=False)
    file_index = 0
    found = False
    while not found:
        file = order[file_index]
        posts = load_list(name=file,location=location)
        ids = []
        for post in posts:
            try:
                rate = post.rate
            except:
                ids.append(post.id)
            else:
                pass
        if len(ids)>0:
            try:
                i = np.random.choice(ids)
                rate = rate_post(posts[i])
                posts[i].rate = rate
                print("==========")
                print(file)
                posts[i].print_post()
                print("----------")
                print("Rating: "+str(rate))
                print("==========")
                save_list(posts,file,location=location)
            except:
                pass
            else:
                found = True
        file_index += 1   


def print_all_rated(location="998_generated\\",
                    limit=20):
    """
    Prints all already rated posts. By default, 
    prints just first couple and the count.

    Parameters
    ----------
    location : str, optional
        The filder with all the posts that may be analyzed. 
        The default is "998_generated\\".
    
    limit : int, optional
        How many first rated posts will be printed. 
        The default if 20.

    Returns
    -------
    None
    """
    total = 0
    posts = load_all_posts(location = location)
    for post in posts:
        try:
            rate = post.rate
        except:
            pass
        else:
            total += 1
            if total<=limit:
                print("==========")
                post.print_post()
                print("----------")
                print("Rating: "+str(rate))
                print("==========")
    print("Rated: "+str(total)+" of "+str(len(posts)))
    print("==========")
    

def patch_negative(location="998_generated\\"):
    """
    Browses through all the posts and patches all the negataive ones 
    that were rated to be better than 0, back down to 0. Not necesary for
    prodution.

    Parameters
    ----------
    location : str, optional
        The filder with all the posts that may be analyzed. 
        The default is "998_generated\\".

    Returns
    -------
    total_fixed : int
        Number of patched posts.
    """
    filelist = os.listdir(location)
    total_fixed = 0
    for file in filelist:
        posts = load_list(name=file,location=location)
        for post in posts:
            try:
                rate = post.rate
            except:
                post.rate = -1
            else:
                if rate!=0 and not post.is_positive:
                    post.rate = 0
                    total_fixed += 1
        save_list(posts,file,location=location)
    return total_fixed
    

def main(n=100):
    print_all_rated()
    
    for i in range(n):
        add_one_rate()
    
    print_all_rated(limit=0)                    
                    

#main(int(sys.argv[1]))
    
    




