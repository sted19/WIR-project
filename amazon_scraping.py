from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests

no_pages = 2

def get_book_genre(book_id):  
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}

    r = requests.get("https://www.amazon.com/dp/{}".format(book_id), headers=headers)
    content = r.content
    soup = BeautifulSoup(content, "html.parser")

    genre = ""

    for d in soup.findAll('div', attrs={'id':'wayfinding-breadcrumbs_feature_div'}):
        res = str(d.select('a')[-1])
        l = res.find('>')
        genre = res[l+1:-4].strip()
        print(genre)

    return genre

if __name__ == "__main__":
    get_book_genre("2070426769")