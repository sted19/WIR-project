# IMPROVING COLLABORATIVE FILTERING THROUGH TOPIC DIVERSIFICATION
## Web Information Retrieval project

### Motivation
Collaborative filtering may tend to suggest items that are very similar to each other. So the authors of [[1]](#1) introduced a metric to assess diversity of a recommendation list and an algorithm that diversifies a given list according to the same metric. <br><br>
They conducted two experiments:
- an offline experiment: measure accuracy reduction derived from diversification,
- an online experiment: measure user satisfaction derived from diversification.

### Our Experiments
We could only conduct the offline experiment because we didn’t have the possibility to reach the users present in our dataset. <br><br>
The authors of the paper used a dataset containing ratings (both implicit and explicit) about books, along with informations about each book.
We retrieved the same dataset from this [website](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). At the same time, informations about topics of the books were missing and we managed to retrieve them from Amazon website through web scraping (the code for the scraping is [here](src/Scraping)). 

We downloaded an additional dataset from [kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset). This dataset contains ratings (explicit only) about movies, along with further informations about each movie. 

From both the dataset we preprocess the data by removing users and items that did not satisfies some properties (here is the code for the data cleaning of the [books](Datasets/BookDatasetCleaning) and here the one of the [movies](/src/movies/organize_movies.ipynb))

#### Accuracy Metrics
We measured different metrics over the two datasets:
For the books:
- Implicit ratings
  - Precision: measured on the entire recommendation list,
  - Recall: measured as precision,
- Explicit ratings 
  - Mean Absolute Error as difference between DCG of optimal ranking and returned one.

For the movies instead
- Explicit ratings 
  - Mean Absolute Error as difference between DCG of optimal ranking and returned one.
 
#### Similarity Measure
The authors used only topic information to compute the similarity measure; this means that two books are different or similar only in terms of their topics. We decided to introduce other elements in the computation of this metric, to obtain more accurate results.

The similarity measure between two books is a weighted sum of the single similarities in terms of topic and author.
The similarity measure between two movies is a weighted sum of the single similarities in terms of genre, production company, production country and original language.



### References 
<a id="1">[1]</a> 
Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, and Georg Lausen. 2005. Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web (WWW ’05). Association for Computing Machinery, New York, NY, USA, 22–32. DOI:https://doi.org/10.1145/1060745.1060754
