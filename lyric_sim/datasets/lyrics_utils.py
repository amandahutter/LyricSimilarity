import numpy as np
import pandas as pd
import sqlite3

def create_similarity_csv(input_path: str, output_path: str, sample=False):
	cnn = sqlite3.connect(input_path)
	df_sim = pd.read_sql_query('SELECT * FROM similars_src', cnn)
	df_sim['target_split'] = df_sim['target'].str.split(',')

    d = {}
	d['Track1'] = []
	d['Track2'] = []
	d['Similarity'] = []

	for i in range(df_sim.shape[0]):
	    targets = df_sim.loc[i, 'target_split']
	    for j in range(0, len(targets), 2):
	        if float(targets[j + 1]) >= 0: 
	            d['Track1'].append(df_sim.loc[i, 'tid'])
	            d['Track2'].append(targets[j])
	            d['Similarity'].append(float(targets[j + 1]))

	similarity_df = pd.DataFrame(d)

	if sample:
		similarity_df = similarity_df.sample(n=1000000, replace=False, random_state=42)

	similarity_df.to_csv(output_path, index = False)

