import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import nltk

if __name__=='__main__' :
	train=pd.read_csv(os.path.join(os.path.dirname(__file__),'data','labeledTrainData.tsv'),header=0,delimiter='\t',quoting=3)
	test=pd.read_csv(os.path.join(os.path.dirname(__file__),'data','testData.tsv'),header=0,delimiter="\t",quoting=3)

print(train)


#removing stopwords and other shit

print("Download text data sets.")
#nltk.download()
clean_train_reviews=[]
print("Cleaning Stuff")
n=len(train['review'])
for i in xrange(0,n):
	val=KaggleWord2VecUtility.review_to_wordlist(train["review"][i],True)
	clean_train_reviews.append(" ".join(val))
	print("-----------------")
	print(str(i))
	print(str(n))
	pass

print(clean_train_reviews)


print("Creating bag of words model...")

vectorizer=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
train_data_features=vectorizer.fit_transform(clean_train_reviews)
train_data_features=train_data_features.toarray()

print("Training the random forest classifier (this may take while)")
forest=RandomForestClassifier(n_estimators=100)
forest=forest.fit(train_data_features,train["sentiment"])

clean_test_reviews=[]

print("Cleaning and parsing the test set movie reviews...")
n=len(test["review"])
for i in xrange(0,n):
	clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i],True)))
	print("-----------------")
	print(str(i))
	print(str(n))
	pass
test_data_features=vectorizer.transform(clean_test_reviews)
test_data_features=test_data_features.toarray()


print("Predicting test labels ...")

result=forest.predict(test_data_features)
output=pd.DataFrame(data={"id":test["id"], "sentiment":result})
#output.to_csv(os.path.join(os.path.dirname(__file__)),'data','Bag_of_words_model.csv',index=False,quoting=3)
print("Wrote results to Bag_of_words model.csv")
print(output)
