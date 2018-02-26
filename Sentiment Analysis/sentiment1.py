from sklearn.datasets import load_files

reviews_train=load_files('data/aclImdb/train/')
text_train, y_train=reviews_train.data,reviews_train.target
print(text_train)
print(y_train)