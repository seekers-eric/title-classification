import pandas as pd
import string, csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *

map_category_index = {'entertainment': 1, 'politics': 2, 'sports': 3, 'technology': 4, 'finance': 5, 'lifestyle': 6}
map_index_category = {v: k for k, v in map_category_index.items()}
news_df = pd.read_csv('./data/training.csv', sep=',')
news_df['CATEGORY'] = news_df.CATEGORY.map(map_category_index)
news_df['TITLE'] = news_df.TITLE.map(
    lambda x: x.lower().translate(str.maketrans('', '', string.punctuation))
)
news_df.head()

TITLES = news_df['TITLE']
CATEGORYS = news_df['CATEGORY']

X_train, X_test, y_train, y_test = train_test_split(
    TITLES,
    CATEGORYS,
    test_size=0.01,
    random_state=1
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

count_vector = CountVectorizer(stop_words='english')
training_data = count_vector.fit_transform(X_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

testing_data = count_vector.transform(X_test)
predictions_single = naive_bayes.predict(testing_data)
predictions_single

print("Accuracy score: ", accuracy_score(y_test, predictions_single))
print("Recall score: ", recall_score(y_test, predictions_single, average='weighted'))
print("Precision score: ", precision_score(y_test, predictions_single, average='weighted'))
print("F1 score: ", f1_score(y_test, predictions_single, average='weighted'))

ts_categories = y_test.values
ts_index = y_test.index.values
diffs = ts_categories - predictions_single
diff_indexes = [i for i, e in np.ndenumerate(diffs) if e != 0]
ts_title = X_test.values


# with open('./data/diff_single.csv', 'w') as csvfile:
#     fieldnames = ['TITLE', 'CATEGORY', 'PREDICTION']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     for x in range(len(diff_indexes)):
#         index = diff_indexes[x]
#         title = ts_title[index].encode('utf8')
#         category = map_index_category[ts_categories[index]]
#         p_category = map_index_category[predictions_single[index]]
#         writer.writerow({'TITLE': title, 'CATEGORY': category, 'PREDICTION': p_category})

# predictions_multi_proba = naive_bayes.predict_proba(testing_data)
# predictions_multi = None
# for x in predictions_multi_proba:
#     top2 = np.array([i + 1 for i in np.argsort(x)[-2:]])
#     if predictions_multi is None:
#         predictions_multi = top2
#     else:
#         predictions_multi = np.vstack((predictions_multi, top2))
#
# with open('./data/diff_multi.csv', 'w') as csvfile:
#     fieldnames = ['TITLE', 'CATEGORY', 'PREDICTION']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     print(predictions_multi.shape[0] - 1)
#     for x in range(predictions_multi.shape[0] - 1):
#         predictions = predictions_multi[x]
#         category = ts_categories[x]
#         if category not in predictions:
#             index = ts_index[x]
#             title = ts_title[x].encode('utf8')
#             predictions = [map_index_category[j] for j in predictions]
#             writer.writerow({'TITLE': title, 'CATEGORY': category, 'PREDICTION': ', '.join(predictions)})

# predictions_multi_proba = naive_bayes.predict_proba(testing_data)
# predictions_multi = None
# for x in predictions_multi_proba:
#     top2 = np.array([i + 1 for i in np.argsort(x)[-2:]])
#     if predictions_multi is None:
#         predictions_multi = top2
#     else:
#         predictions_multi = np.vstack((predictions_multi, top2))
#
# with open('./data/test_case_multi.csv', 'w') as csvfile:
#     fieldnames = ['TITLE', 'CATEGORY', 'PREDICTION', 'STATUS']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     print(predictions_multi.shape[0] - 1)
#     for x in range(predictions_multi.shape[0] - 1):
#         predictions = predictions_multi[x]
#         category = ts_categories[x]
#         status = True
#         if category not in predictions:
#             status = False
#         index = ts_index[x]
#         title = ts_title[x].encode('utf8')
#         predictions = reversed([map_index_category[j] for j in predictions])
#         writer.writerow({'TITLE': title, 'CATEGORY': map_index_category[category], 'PREDICTION': ', '.join(predictions), 'STATUS': status})


print("\nWelcome to the news title classification. Tell me your news")

choice = ''

while choice != 'quit':
    choice = input("\nPlease tell me someone I should know, or enter 'quit': ")
    i = count_vector.transform([choice])
    cs = naive_bayes.predict_proba(i)
    print(cs)
    top2 = reversed([i + 1 for i in np.argsort(cs)[-2:][::-1]][0][-2:])
    top2_c = [map_index_category[x] for x in top2]
    single_c = naive_bayes.predict(i)[0]
    c = map_index_category[single_c]
    print(top2_c)
    # print(c)
