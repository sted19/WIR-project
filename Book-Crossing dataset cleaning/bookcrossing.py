import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

# Ensures compliance with the constraints
def integrity(table):
    number_of_iterations = 0
    while (not(table[table.groupby('userId')['bookTitle'].transform('size') < 5].empty and table[table.groupby('bookTitle')['userId'].transform('size') < 20].empty)):
        table = table[table.groupby('userId')['bookTitle'].transform('size') >= 5]
        table = table[table.groupby('bookTitle')['userId'].transform('size') >= 20]
        number_of_iterations += 1
    print(f'Number of iterations needed to respect the constraints: {number_of_iterations}')
    print('\n')
    return table
    

if __name__ == "__main__":
    
    ratings = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
    ratings.columns = ['userId', 'ISBN', 'bookRating']

    users = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Users.csv', delimiter=";", encoding="latin1")
    users.columns = ['userId', 'location', 'age']

    books = pd.read_csv('../Datasets/BX-CSV-Dump/BX-Books.csv', delimiter=";", encoding="latin-1", error_bad_lines=False)
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL']
    print('\n')
    print(books.head())

    print("Shapes of the 3 tables:")
    print(ratings.shape)
    print(users.shape)
    print(books.shape)
    print('\n')
    
    # We don't need images column in Books
    # For the moment we leave all other infos for books, but maybe some of them will be useless
    books.drop(['imageURLS','imageURLM','imageURLL'], axis=1, inplace=True)

    # We don't need location and age of the users
    users.drop(['location', 'age'], axis=1, inplace=True)

    print("Distinct values of rating in the ratings table:")
    print(ratings['bookRating'].unique())
    print('\n')

    # The Types are mixed in books.yearOfPublication, between Integers and Strings... Must set one type.
    # Then there are wrong Data, in that they have set Publisher instead of year of publication 'Dk Publishing' and ''Gallimard'.
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

    print("Distinct values of year in books tables:")
    print(sorted(books['yearOfPublication'].unique()))
    print('\n')
    # There are a lot of books with yearOfPublication == 0
    print("Books with year = 0")
    print(books.loc[books.yearOfPublication == 0,:])
    print('\n')

    # create df of old books
    historical_books = books[(books.yearOfPublication>0) & (books.yearOfPublication<1900)]
    hist_books_mini = historical_books[['bookTitle', 'yearOfPublication']]
    print(f'Historical books:\n{hist_books_mini}')
    print('\n')
    print(f'Length of books dataset before removal: {len(books)}')
    # remove historical books
    books = books.loc[~(books.ISBN.isin(historical_books.ISBN))]
    print(f'Length of books dataset after removal: {len(books)}')

    # Here we could remove all the "future" (the dataset is from 2004) and year = 0 books, or set the year of them to NaN
    #books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN

    # We clean up the ampersand formatting in the Publisher field
    books.publisher = books.publisher.str.replace('&amp', '&', regex=False)

    # Check that there are no duplicated book entries
    uniq_books = books.ISBN.nunique()
    all_books = books.ISBN.count()
    print(f'No. of unique books: {uniq_books} | All book entries: {all_books}')
    print('\n')

    # Check for empty or NaN values
    empty_string_publisher = books[books.publisher == ''].publisher.count()
    nan_publisher = books.publisher.isnull().sum()
    print(f'There are {empty_string_publisher} entries with empty strings, and {nan_publisher} NaN entries in the Publisher field')
    print('\n')
    empty_string_author = books[books.bookAuthor == ''].bookAuthor.count()
    nan_author = books.bookAuthor.isnull().sum()
    print(f'There are {empty_string_author} entries with empty strings, and {nan_author} NaN entries in the Author field')
    print('\n')

    # Natural join between books and ratings
    # We obtain the ratings of existing books
    books_ratings = pd.merge(books, ratings, on='ISBN')
    # Natural join between books_ratings and users
    # We obtain the ratings of existing books expressed by existing users
    users_books_ratings = pd.merge(users, books_ratings, on='userId')
    users_books_ratings = integrity(users_books_ratings)
    print(f'In the ratings table we have {len(users_books_ratings.index)} ratings expressed by {users_books_ratings.userId.nunique()} different users, associated to {users_books_ratings.ISBN.nunique()} different ISBNs of {users_books_ratings.bookTitle.nunique()} different book titles')
    print('\n')
    print("We have several ISBNs associated to a single book title:")
    print('\n')
    print(users_books_ratings.groupby('bookTitle').ISBN.nunique().sort_values(ascending=False)[:10])
    print('\n')
    multiple_isbns = users_books_ratings.groupby('bookTitle').ISBN.nunique()
    print(multiple_isbns.value_counts())
    print('\n')
    has_mult_isbns = multiple_isbns.where(multiple_isbns > 1)
    # remove NaNs, which in this case is books with a single ISBN number
    has_mult_isbns.dropna(inplace = True)
    print(f'There are {len(has_mult_isbns)} book titles with multiple ISBN numbers which we will try to re-assign to a unique identifier')
    print('\n')
    # Create dictionary for books with multiple isbns
    '''
    def make_isbn_dict(df):
        title_isbn_dict = {}
        for title in multiple_isbns.index:
            isbn_series = df.loc[df.bookTitle==title].ISBN.unique() # returns only the unique ISBNs
            title_isbn_dict[title] = isbn_series.tolist()
        return title_isbn_dict
    
    dict_unique_isbn = make_isbn_dict(users_books_ratings)
    
    # As the loop takes a while to run (few min), pickle this dict for future use
    with open('multiple_isbn_dict.pickle', 'wb') as handle:
        pickle.dump(dict_unique_isbn, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    
    # LOAD isbn_dict back into namespace
    with open('multiple_isbn_dict.pickle', 'rb') as handle:
        multiple_isbn_dict = pickle.load(handle)

    # Add 'unique_isbn' column to the table dataframe that includes the first ISBN if multiple ISBNS,
    # or just the ISBN if only 1 ISBN present anyway.
    def add_unique_isbn_col(df):
        df['uniqueISBN'] = df.apply(lambda row: multiple_isbn_dict[row.bookTitle][0], axis=1)
        return df

    users_books_ratings = add_unique_isbn_col(users_books_ratings)

    # topics = dictionary where they keys are the several ISBNs and the values are the product category obtained from Amazon.com
    f = open('topics.json')
    data = json.load(f)
    f.close()
    topics = {}
    for value in data:
        topics[value['product_asin']] = value['product_category']
    
    # We assign the right topic to each row
    users_books_ratings['topic'] = users_books_ratings.apply(lambda row: topics[str(row.uniqueISBN)] if (str(row.uniqueISBN)) in topics.keys() else "", axis=1)
    # We remove the ratings relative to books for which we were not able to get the topic
    users_books_ratings.drop(users_books_ratings[users_books_ratings.topic == ""].index, inplace=True)
    print(f'After removing  the books without topic, in the ratings table we have {len(users_books_ratings.index)} ratings expressed by {users_books_ratings.userId.nunique()} different users, associated to {users_books_ratings.ISBN.nunique()} different ISBNs of {users_books_ratings.bookTitle.nunique()} different book titles')
    print('\n')
    # Since we have removed some infos, we have to check again the integrity and eventually modify the table
    users_books_ratings = integrity(users_books_ratings)
    print(f'After the last integrity check, in the ratings table we have {len(users_books_ratings.index)} ratings expressed by {users_books_ratings.userId.nunique()} different users, associated to {users_books_ratings.ISBN.nunique()} different ISBNs of {users_books_ratings.bookTitle.nunique()} different book titles')
    print('\n')

    # Check missing values
    users_books_ratings.info()
    print('\n')

    explicit_ratings = users_books_ratings[users_books_ratings.bookRating != 0]
    implicit_ratings = users_books_ratings[users_books_ratings.bookRating == 0]
    print("Shapes of ratings tables:")
    print(users_books_ratings.shape)
    print(explicit_ratings.shape)
    print(implicit_ratings.shape)
    print('\n')

    print(f'No. of unique book titles in only explicit_ratings: {explicit_ratings.bookTitle.nunique()}')
    print(f'No. of unique book titles in only implicit_ratings: {implicit_ratings.bookTitle.nunique()}')
    print('\n')

    print(f'No. of unique ISBNs in explicit_ratings: {explicit_ratings.ISBN.nunique()}')
    print(f'No. of unique ISBNs in implicit_ratings: {implicit_ratings.ISBN.nunique()}')
    print('\n')

    print(f'No. of unique users in explicit_ratings: {explicit_ratings.userId.nunique()}')
    print(f'No. of unique users in implicit_ratings: {implicit_ratings.userId.nunique()}')
    print('\n')

    users_books_ratings.to_csv('All_ratings.csv', sep='\t')
    explicit_ratings.to_csv('Explicit.csv', sep='\t')
    implicit_ratings.to_csv('Implicit.csv', sep='\t')

    '''
    # Create a dictionary where the keys are the several uniqueISBNs in users_books_ratings and the values are lists of ['uniqueISBN', 'bookTitle', 'bookAuthor', 'topic']
    books_dict = {}

    for isbn in users_books_ratings.uniqueISBN.unique():
        rslt_df = users_books_ratings[users_books_ratings['ISBN'] == isbn]
        values = list([tuple(x) for x in rslt_df.values][0])
        topic = str(values[-1])
        values = values[1:4]
        values.append(topic)
        books_dict[str(isbn)] = values

    with open('books_dict.pickle', 'wb') as handle:
        pickle.dump(books_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''