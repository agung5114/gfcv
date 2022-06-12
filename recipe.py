import requests
from recipe_scrapers import scrape_html, scrape_me
import re
from pygsearch import gsearch

def searchLink(foodname):
    search = gsearch(f"{foodname} site=allrecipes.com",1)
    result1 = str(search.results[0])
    text = result1.split("link=",1)[1]
    link = text.split(">",1)[0]
    return link[1:-2]

rd = searchLink('rendang')
# print(rd)

def scrapeRecipe(url):
    # url = f"https://www.allrecipes.com/recipe/72567/panna-cotta/"
    html = requests.get(url).content

    scraper = scrape_html(html=html, org_url=url)
    # title = scraper.title()
    # total_time=scraper.total_time()
    # yields= scraper.yields()
    ingredients =scraper.ingredients()
    ingredient = []
    for i in ingredients:
        ingredient.append(re.sub (r'([^a-zA-Z ]+?)', '', i))

    instructions = scraper.instructions()
    # links = scraper.links()
    nutrients = scraper.nutrients()

    return {'ingredient':ingredient,'instructions':instructions}

# pcota =scrapeRecipe('https://www.allrecipes.com/recipe/72567/panna-cotta/')
pcota =scrapeRecipe(rd)
print(pcota)