from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import pandas as pd

no_pages = 2

def get_book_genre(book_id):  
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}

    r = requests.get("https://www.amazon.com/dp/{}".format(book_id), headers=headers)
    content = r.content
    soup = BeautifulSoup(content, "html.parser")
    print(soup)

    genre = ""

    for d in soup.findAll('div', attrs={'id':'wayfinding-breadcrumbs_feature_div'}):
        res = str(d.select('a')[-1])
        l = res.find('>')
        genre = res[l+1:-4].strip()
        print(genre)

    return genre

if __name__ == "__main__":
    books = pd.read_csv('./Datasets/BX-CSV-Dump/BX-Books.csv', delimiter=";", encoding="latin-1", error_bad_lines=False)
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL']
    books.drop(['imageURLS','imageURLM','imageURLL'], axis=1, inplace=True)
    historical_books = books[(books.yearOfPublication>0) & (books.yearOfPublication<1900)]
    hist_books_mini = historical_books[['bookTitle', 'yearOfPublication']]
    print(f'Historical books:\n{hist_books_mini}')
    print('\n')
    print(f'Length of books dataset before removal: {len(books)}')
    # remove historical books
    books = books.loc[~(books.ISBN.isin(historical_books.ISBN))]
    print(f'Length of books dataset after removal: {len(books)}')
    books.info()
    #books['topic'] = books.apply(lambda row: get_book_genre(row.ISBN), axis=1)