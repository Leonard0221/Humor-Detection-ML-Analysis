import pandas as pd
import random
from gensim.models import Word2Vec
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

humour = pd.read_pickle('humorous_oneliners.pickle')
proverb = pd.read_pickle('proverbs.pickle')
wiki = pd.read_pickle('wiki_sentences.pickle')
long_humour = pd.read_pickle('oneliners_incl_doubles.pickle')
reuters = pd.read_pickle('reuters_headlines.pickle')
print(len(humour), len(proverb), len(wiki), len(reuters))
humour_record = [(line, 1) for line in humour]
proverb_record = [(line, 0) for line in proverb]
wiki_record = [(line, 0) for line in wiki]
reuter_record = [(line, 0) for line in reuters]

jokes_real = humour_record
non_jokes = proverb_record + wiki_record + reuter_record
columns = ['sentence', 'class']
random.shuffle(non_jokes)
non_jokes = non_jokes[:len(jokes_real)]
df_record = jokes_real + non_jokes
df = pd.DataFrame(df_record, columns=columns)
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())
df['class'].value_counts()
text = []
for sentence, classification in df_record:
    sent_word_list = [word for word in sentence.lower().split()]
    text.append(sent_word_list)

Word2Vectors = Word2Vec(text, min_count=1)
vectors = []
for i in range(len(df['sentence'])):
    pass_list = df['sentence'][i]
    if len(pass_list) != 0:
        pass_vector = sum([Word2Vectors.wv[w] for w in pass_list.lower().split()]) / (len(pass_list.split()) + 0.00001)
    else:
        pass_vector = np.zeros((100,))
    vectors.append(pass_vector)

X = pd.DataFrame(vectors, columns=range(100))
print(X.head())
print(X.shape)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print('Training Data: ', X_train.shape, y_train.shape)
print('Test Data: ', X_test.shape, y_test.shape)

svm = SVC(kernel='linear', random_state=1, C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print()
print('1. We use SVM: ')
print('Test Accuracy Score: ', accuracy_score(y_pred, y_test))

scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

svm = SVC(kernel='poly', coef0=1, degree=3)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print()
print('1. We use SVM: ')
print('Test Accuracy Score: ', accuracy_score(y_pred, y_test))

scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


svm = SVC(kernel='rbf', gamma=0.7)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print()
print('1. We use SVM: ')
print('Test Accuracy Score: ', accuracy_score(y_pred, y_test))

scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print()
print('2. We use Logistic Regression: ')
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print('Test Accuracy score: ', accuracy_score(y_pred, y_test))
scores = cross_val_score(estimator=lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print()
print('3. We use Decision Tree: ')
tree = DecisionTreeClassifier(random_state=0, max_depth=8)
tree = tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print('Test Accuracy score: ', accuracy_score(y_pred, y_test))
scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10, n_jobs=-1)
print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

