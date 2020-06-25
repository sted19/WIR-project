import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    
    ratings = pd.read_csv('./Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
    ratings.columns = ['userId', 'ISBN', 'bookRating']

    users = pd.read_csv('./Datasets/BX-CSV-Dump/BX-Users.csv', delimiter=";", encoding="latin1")
    users.columns = ['userId', 'location', 'age']

    books = pd.read_csv('./Datasets/BX-CSV-Dump/BX-Books.csv', delimiter=";", encoding="latin-1", error_bad_lines=False)
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL']

    print("Shapes of the 3 tables:")
    print(ratings.shape)
    print(users.shape)
    print(books.shape)
    print('\n')
    
    #We don't need images column in Books
    books.drop(['imageURLS','imageURLM','imageURLL'], axis=1, inplace=True)

    print("Distinct values of rating in the ratings table:")
    print(ratings['bookRating'].unique())
    print('\n')

    #The Types are mixed in books.yearOfPublication, between Integers and Strings... Must set one type. Then there are wrong Data, in that they have set Publisher instead of year of publication 'Dk Publishing' and ''Gallimard'.
    print("Year and publisher reversed:")
    print(books.loc[books.yearOfPublication == 'DK Publishing Inc',:])
    books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
    books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
    books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
    books.loc[books.ISBN == '0789466953', 'yearOfPublication'] = 2000
    books.loc[books.ISBN == '0789466953', 'bookAuthor'] = "James Buckley"
    books.loc[books.ISBN == '0789466953', 'publisher'] = "DK Publishing Inc"
    print(books.loc[books.yearOfPublication == 'Gallimard'])
    books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
    books.loc[books.ISBN == '2070426769','bookAuthor'] = 'Jean-Marie Gustave Le ClÃ?Â©zio'
    books.loc[books.ISBN == '2070426769','publisher'] = 'Gallimard'
    books.yearOfPublication = pd.to_numeric(books.yearOfPublication)
    print('\n')
    
    print("Distinct values of year in books tables:")
    print(sorted(books['yearOfPublication'].unique()))
    print('\n')
    
    # There are a lot of books with yearOfPublication == 0
    print("Books with year = 0")
    print(books.loc[books.yearOfPublication == 0,:])
    print('\n')

    # Merging between tables
    rating_book = pd.merge(ratings, books, on='ISBN')
    all_ratings = pd.merge(rating_book, users, on='userId')

    # Some visual understanding of the Data
    sns.countplot(all_ratings.bookRating)
    plt.show()
    
    #Segragating implicit and explict ratings datasets (the explicits are the ones with integer votes in [1, 10], the implicits are the ones with vote = 0)
    ratings_explicit = all_ratings[all_ratings.bookRating != 0]
    ratings_implicit = all_ratings[all_ratings.bookRating == 0]
    print("Shapes of ratings tables:")
    print(all_ratings.shape)
    print(ratings_explicit.shape)
    print(ratings_implicit.shape)


    