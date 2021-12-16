typical config for these models will change batch_size, nothing else
batch_size = 128 is default
try 64, 256, 512 
train_lstm_all_params.py
accuracy_lstm_all_params.py
---
  lyrics_db_path: './data_files/mxm_lyrics_cleaned.db'
  mxm_db_path: './data_files/mxm_dataset.db'
  lastfm_db_path: './data_files/mxm_lastfm.db'
  batch_size: 128
  emb_size: 20
  hidden_size: 10
  dropout: 0.9
  num_fc: 2
  combo_unit: 'ALL'
  num_epochs: 5
  num_workers: 0
  num_layers: 2
  dropout_first : False
  learning_rate: 0.001
  momentum: 0.90
  model_name: 'lstm_model_fc2_all_cleaned_2l_256b'