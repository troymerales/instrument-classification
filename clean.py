import pandas as pd

df = pd.read_csv('dataset/openmic-2018/openmic-2018-aggregated-labels.csv')
string_ins = ['banjo', 'bass', 'cello', 'guitar', 'mandolin', 'ukulele', 'violin']

df = df[df['instrument'].isin(string_ins)]

print(df.head())

counts = df['instrument'].value_counts()

print(counts)

