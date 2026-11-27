# Sentiment Analysis of Restaurant Reviews (IOI City Mall)

End-to-end NLP pipeline on Google Maps reviews of five restaurants in IOI City Mall, Putrajaya, Malaysia: scrape the reviews, clean the text, classify sentiment, find the topics customers talk about, and present it in a Power BI dashboard for restaurant managers. Text Analytics course project at UNITEN.

## Pipeline

1. **Scraping**: Selenium and BeautifulSoup collect reviews and star ratings from Google Maps (2,577 reviews across KyoChon 1991, Ombak Kitchen, Mr Dakgalbi, Nando's and The Manhattan Fish Market).
2. **Preprocessing**: NLTK tokenisation, stop-word removal and lemmatisation.
3. **Sentiment**: lexicon-based classification with **TextBlob** and **VADER**, evaluated against labels derived from each review's star rating.
4. **Topic modelling**: TF-IDF and **LDA** (scikit-learn), with topics grouped into Food Quality, Customer Service and Restaurant Condition.
5. **Dashboard**: Power BI visualisation of sentiment and topics per restaurant.

## Results

| Model | Accuracy | Macro F1 | Negative F1 | Positive F1 |
|---|---|---|---|---|
| TextBlob | 0.80 | 0.55 | 0.69 | 0.89 |
| VADER | 0.80 | 0.55 | 0.64 | 0.89 |

Both lexicons handle clearly positive reviews well but struggle with neutral ones (F1 under 0.15), since 3-star reviews often mix praise and complaints. A fine-tuned transformer (for example DistilBERT) would be the natural next step.

## Dashboard

![Power BI dashboard 1](https://github.com/NFL-Nujaimin/Sentiment-Analysis-of-IOI-Restaurant-Review/assets/126408968/8ac2691b-f63f-4fb8-b9b2-4d3c6433063f)

![Power BI dashboard 2](https://github.com/NFL-Nujaimin/Sentiment-Analysis-of-IOI-Restaurant-Review/assets/126408968/451fa4b1-0728-4769-8518-e23d6fb492db)

![Power BI dashboard 3](https://github.com/NFL-Nujaimin/Sentiment-Analysis-of-IOI-Restaurant-Review/assets/126408968/e195e3c4-41b2-4328-9d61-a389be268f15)

![Power BI dashboard 4](https://github.com/NFL-Nujaimin/Sentiment-Analysis-of-IOI-Restaurant-Review/assets/126408968/e0c96dbc-9c46-45de-9587-43ba2bc02593)

![Power BI dashboard 5](https://github.com/NFL-Nujaimin/Sentiment-Analysis-of-IOI-Restaurant-Review/assets/126408968/4c53d196-1849-4e1e-805b-3586ae542049)

Presentation video: https://youtu.be/_S_sW6lxNY4

## Files

| File | What it is |
|---|---|
| `sentiment_analysis.ipynb` | Scraping, preprocessing, sentiment analysis and topic modelling |
| `restaurant_reviews.csv` | Collected reviews with processed text, predicted sentiment and topic |

## Run it

```bash
pip install pandas nltk textblob vaderSentiment scikit-learn selenium beautifulsoup4 jupyter
jupyter notebook sentiment_analysis.ipynb
```

The scraping cell needs Chrome and ChromeDriver. To skip it, start from the preprocessing section, which reads `restaurant_reviews.csv`.

Tools: Python, Selenium, BeautifulSoup, NLTK, TextBlob, VADER, scikit-learn, Power BI.
