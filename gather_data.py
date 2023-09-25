import urllib.request
import datetime as dt
import pickle
import os

class Post:
    
    def __init__(self, datetime, city, is_positive, comment):
        self.datetime = datetime
        self.city = city
        self.is_positive = is_positive
        self.comment = comment
        
    def print_post(self):
        print(self.datetime.strftime("%d.%m.%Y")+
              " @ "+ self.city+
              " - Rastli"*self.is_positive+" - Nerastli"*(1-self.is_positive)+
              "\n "+ self.comment)


def url(year,month,region=0,page=0):
    """
    To construct the proper URL of shroom growing archive.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    region : int, optional
        Region ID. The default is 0, which stand for all regions.
    page : int, optional
        Page number. The default is 0, which is the first page.

    Returns
    -------
    str
        The URL.

    """
    return "https://www.nahuby.sk/vyskyt_hub_vyhladavanie.php?"+ \
            "year="+str(year)+ \
            "&month="+str(month)+ \
            "&region_id="+str(region)+ \
            "&users=&mode=&page="+str(page)
            
            
def fetch_code_from_url(url):
    """
    To return the webpage source code.

    Parameters
    ----------
    url : str
        The target URL.

    Returns
    -------
    str
        The code.

    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers={'User-Agent':user_agent,} 

    request=urllib.request.Request(url,None,headers) #The assembled request
    response = urllib.request.urlopen(request)
    return response.read().decode("utf-8")


def get_page_count(decoded_source_html):
    """
    Get the number of pages to go through.

    Parameters
    ----------
    decoded_source_html : str
        The source code of the page.

    Returns
    -------
    page_count : int
        The page count.

    """
    if decoded_source_html.find("Počet")>-1:
        start = decoded_source_html.find("Počet")+7
        end = decoded_source_html.find("</td>",start)
        posts_count = int(decoded_source_html[start:end])
        page_count = posts_count//50+1
        return page_count
    else:
        return 0


def extract_posts(decoded_source_html):
    """
    Extract useful data from a archive report forum webpage.

    Parameters
    ----------
    decoded_source_html : str
        Decoded source code of the webpage.

    Returns
    -------
    list_of_posts : list of object Post
        A list of records ectracted from the page.

    """
    list_of_posts = []
    start_character = 0
    while decoded_source_html.find(
            "spravy-o-raste-hub spravy-o-raste-hub-",
            start_character) > -1:
        
        next_index = decoded_source_html.find(
            "spravy-o-raste-hub spravy-o-raste-hub-",
            start_character)
        
        #report positive or negative
        is_positive = decoded_source_html[next_index+38] == "y"
        
        #date of the report
        date_index = decoded_source_html.find("datum-normal",next_index)
        day = int(decoded_source_html[date_index+17:date_index+19])
        month = int(decoded_source_html[date_index+21:date_index+23])
        year = int(decoded_source_html[date_index+36:date_index+40])
        datetime = dt.datetime(year,month,day)
        
        #report city
        start = decoded_source_html.find("</b> v roku",next_index)+18
        end = decoded_source_html.find("- huby",start)-1
        city = decoded_source_html[start:end]
        
        #report text
        start = decoded_source_html.find("komentar",next_index)+10
        end = decoded_source_html.find("</p>",start)
        comment = decoded_source_html[start:end]
        
        #append_post
        list_of_posts.append(Post(datetime,city,is_positive,comment))
        
        start_character = next_index+1
        
    return list_of_posts
    

def get_all_posts(year,month,region=0):
    """
    The function to get all postf from all the pages for a date.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    region : int, optional
        ID of the region. The default is 0, which means all regions.

    Returns
    -------
    all_posts : TYPE
        DESCRIPTION.

    """
    page_zero_str = fetch_code_from_url(url(year,month))
    page_count = get_page_count(page_zero_str)
    
    all_posts = []
    for page in range(page_count):
        all_posts += extract_posts(
                        fetch_code_from_url(
                            url(year,month,region,page)))
    
    return all_posts


def save_list(data,name,location="998_generated\\"):
    with open(location+name, "wb") as f:  
        pickle.dump(data, f)
        
        
def load_list(name,location="998_generated\\"):
    with open(location+name, "rb") as f:
        data = pickle.load(f)
    return data


def mine_years(start_year=2004,end_year=2023):
    for year in range(start_year,end_year+1):
        for month in range(1,12+1):
            posts = get_all_posts(year,month)
            save_list(posts,str(year)+"_"+str(month)+".pkl")


def load_all_posts(location="998_generated\\"):
    all_posts = []
    for file in os.listdir(location):
        posts = load_list(file,location=location)
        for post in posts:
            all_posts.append(post)
    return all_posts
    


#posts = load_all_posts() 






















