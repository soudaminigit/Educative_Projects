import pandas as pd
import matplotlib as plt
train_data= pd.read_csv('../Dataset/IMDB_Review.csv')
train_data.head(5)
train_data.info()
review_list = train_data['sentiment']
print("Records with negative sentiment")
print(review_list.value_counts()['negative'])
print("Records with positive sentiment")
print(review_list.value_counts()['positive'])
#print("Sample Review")
#print(train_data['review'][0])