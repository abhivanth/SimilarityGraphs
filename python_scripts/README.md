# Datasets
text datasets chosen for initial analysis
## News Category DataSet : 

This dataset contains around 210k news headlines from 2012 to 2022 from HuffPost. This is one of the biggest news datasets and can serve as a benchmark for a variety of computational linguistic tasks. HuffPost stopped maintaining an extensive archive of news articles sometime after this dataset was first collected in 2018, so it is not possible to collect such a dataset in the present day. Due to changes in the website, there are about 200k headlines between 2012 and May 2018 and 10k headlines between May 2018 and 2022.

Each record in the dataset consists of the following attributes:

- category: category in which the article was published.
- headline: the headline of the news article.
- authors: list of authors who contributed to the article.
- link: link to the original news article.
- short_description: Abstract of the news article.
- date: publication date of the article. 

There are a total of 42 news categories in the dataset.
Source : Misra, Rishabh. "News Category Dataset.". [Misra 2022](https://www.researchgate.net/publication/363843926_News_Category_Dataset)

**newsdata_set.csv** is vectorized dataset with 20000 datapoints with of news description to their corresponding news categories. 
The vectorization technique used is **TF-IDF (Term Frequency-Inverse Document Frequency)** from **scikit-learn**

## Multidomain sentiment analysis dataset

This dataset consists of 8000 product reviews obtained from amazon.com where the products are books, dvd, electronics and kitchen. 
There are 1000 positive reviews and 1000 negative reviews for each of the four product domains. The dataset is processed into
two based on sentiment and based on stars.

Source : John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. Association of Computational Linguistics (ACL), 2007.[Blitzer 2007](https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf)

## Next Steps

- To vectorize the Multidomain Sentiment Dataset.
- To use different vetorization methods to obtain a differnt perspective of the datasets.
- Different methods include Spacy lib,Latent Dirichlet Allocation,BERT or BART summarizer


