# Social Media Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings and Recommendations](#key-findings-and-recommendations)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [License](#license)
- [Contact](#contact)
- [Contributing](#contributing)

## Project Overview
This project conducts a **sentiment analysis** on a dataset of over 52,000 tweets to uncover public sentiment trends, patterns, and actionable insights. Leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques, it analyzes tweet sentiments using tools like VADER, TextBlob, BERT, and an AdaBoost Classifier, alongside topic modeling with Latent Dirichlet Allocation (LDA). The project aims to assist companies, policymakers, and individuals in understanding social media dynamics.

### Objectives
- Analyze sentiment distribution (positive, neutral, negative) in tweets.
- Identify key topics and themes using topic modeling.
- Provide actionable recommendations based on sentiment and engagement insights.

---

## Dataset
The dataset comprises **52,542 tweets** collected from Twitter, stored in `tweets.csv` under the `Dataset/` directory.

### Key Features
- **Size**: 52,542 records
- **Columns**:
  - `author`: Twitter username.
  - `content`: Tweet text.
  - `country`: Country of origin (mostly missing).
  - `date_time`: Timestamp of the tweet.
  - `language`: Language (predominantly English).
  - `number_of_likes`, `number_of_shares`: Engagement metrics.
  - Additional fields: `id`, `latitude`, `longitude` (sparse).

### Data Characteristics
- **High Cardinality**: Unique `content` and `author` values.
- **Missing Values**: Sparse `country`, `latitude`, and `longitude`.
- **Class Imbalance**: Neutral sentiments dominate.

---

## Methodology

### Data Preprocessing
1. **Text Cleaning**:
   - Removed URLs, hashtags, mentions, and special characters.
   - Applied tokenization and lemmatization.
2. **Stopword Removal**: Eliminated non-informative words.
3. **Feature Engineering**: Added `cleaned_text` and `processed_text` columns.

### Sentiment Analysis Techniques
1. **VADER**: Polarity-based scoring for short text (70% accuracy).
2. **TextBlob**: Polarity and subjectivity scoring (lower recall for negatives).
3. **BERT**: Contextual sentiment classification (39.9% accuracy, needs tuning).

### Machine Learning
- **Model**: AdaBoost Classifier with TF-IDF vectorization.
- **Performance**:
  - Before Tuning: 80.5% accuracy.
  - After Tuning (150 estimators, 1.0 learning rate, 3,000 max features): 83.9% accuracy.

### Topic Modeling
- **Method**: Latent Dirichlet Allocation (LDA).
- **Topics Identified**: 8 topics (e.g., gratitude, social events, complaints).

### Tools and Libraries
- **Python Libraries**:
  - `pandas`, `numpy`: Data manipulation.
  - `nltk`, `textblob`, `transformers`: NLP and sentiment analysis.
  - `scikit-learn`, `gensim`: ML and topic modeling.
  - `matplotlib`, `seaborn`, `wordcloud`: Visualizations.

### Visualizations
- Sentiment distribution (bar/pie charts).
- Confusion matrices for AdaBoost (before/after tuning).
- Word clouds and heatmaps for topic modeling.

---

## Key Findings and Recommendations

### Key Findings
1. **Sentiment Distribution**: Neutral tweets dominate, followed by positive and negative.
2. **Engagement**: Positive tweets correlate with higher likes and shares.
3. **Topics**: Gratitude, entertainment, and complaints are prevalent themes.
4. **Model Performance**: AdaBoost (83.9%) outperforms VADER (70%) and BERT (39.9%).

### Recommendations
1. **Leverage Positive Sentiment**: Use optimistic tweets for marketing campaigns.
2. **Address Negative Feedback**: Monitor and respond to complaints promptly.
3. **Improve Models**: Fine-tune BERT and handle class imbalance with oversampling or class weights.
4. **Targeted Engagement**: Focus on neutral tweets with tailored content to boost interaction.

---

## Installation

### Prerequisites
- Python 3.8+
- Required libraries (see `requirements.txt`).

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Parmod2310/Social-Media-Sentiment-Analysis.git
   cd Social-Media-Sentiment-Analysis
   ```
2.Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK resources
   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```
4. Place the dataset:
- Ensure `tweets.csv` is in the `Dataset/` directory.

---

## Usage

### Running the Analysis
Run the main script (assuming a consolidated script exists, e.g., `Social Media Sentiment Analysis.ipynb`):
```bash
python Social Media Sentiment Analysis.ipynb
```
- This loads the dataset, preprocesses text, performs sentiments analysis, trains the AdaBoost model, and generates visualizations.

### Example: Sentiment Visualization
To Visualize sentiment distribution

```python
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('Dataset/tweets.csv')
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['content'].apply(lambda x: 'positive' if sid.polarity_scores(x)['compound'] > 0.05 else 'negative' if sid.polarity_scores(x)['compound'] < -0.05 else 'neutral')
df['sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution')
plt.show()
```

### Example: Confusion Matrix
To plot the AdaBoost confusion matrix:

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Assuming y_test, y_pred_tuned are available from model training
sns.heatmap(confusion_matrix(y_test, y_pred_tuned), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - AdaBoost (Tuned)')
plt.show()
```
---
## References
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, reach out via [p921035@gmail.com](mailto:p921035@gmail.com).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.
  
