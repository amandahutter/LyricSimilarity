import sqlite3
import argparse
from sqlite3 import Error as SqlException
import os
import time
import swagger_client
from swagger_client.rest import ApiException
from typing import Tuple, Union

# str | Account api key, to be used in every api call
swagger_client.configuration.api_key['apikey'] = os.getenv('MXM_API_KEY')

LYRICS_DB = "./files/mxm_lyrics.db"
MXM_TRAIN_FILE = "./files/mxm_dataset_train.txt"
MXM_TEST_FILE = "./files/mxm_dataset_test.txt"

def main():
    parser = argparse.ArgumentParser(description="Download lyrics from mxm")
    parser.add_argument('--wait', help="How many seconds to wait between requests in floating point seconds.", default=.1)
    args = parser.parse_args()
    download_and_insert_lyrics(MXM_TRAIN_FILE, wait=args.wait)

def download_and_insert_lyrics(filepath, wait) -> None:
    conn = None
    try:
        conn = sqlite3.connect(LYRICS_DB)
        create_lyrics_table(conn)
        read_mxm_file_and_download_lyrics(conn, filepath, wait)
    except SqlException as sqlex:
        print("Sqlite exception:")
        print(sqlex)
    finally:
        if conn:
            conn.close()

def read_mxm_file_and_download_lyrics(conn, filepath, wait) -> None:
    with open(filepath) as inputfile:
        for line in inputfile:
            if line[0] is '#' or line[0] is '%':
                continue
            else:
                values = line.split(',')
                mxm_id = values[1]
                lyrics_response = download_lyrics_throttled(mxm_id, wait)
                if lyrics_response:
                    insert_lyrics(conn, mxm_id, lyrics_response)

def create_lyrics_table(conn) -> None:
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS lyrics (
        mxm_id INTEGER PRIMARY KEY,
        lyrics_id INTEGER,
        lyrics TEXT,
        EXPLICIT INTEGER,
        lyrics_language TEXT,
        lyrics_language_description TEXT
    )
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
    except SqlException as sqlex:
        print("Error creating lyrics table.")
        print(sqlex)
        exit()

def download_lyrics_throttled(mxm_id, wait) -> Union[dict, None]:
    time.sleep(wait)
    return download_lyrics(mxm_id)

def download_lyrics(mxm_id) -> Union[dict, None]:
    lyrics_api = swagger_client.LyricsApi()
    try:
        lyrics_response = lyrics_api.track_lyrics_get_get(mxm_id)
    except ApiException as api_ex:
        print("Error sending request to musix match.")
        print(api_ex)
    if (lyrics_response.message.header.status_code != 200):
        print(f'Failed to download lyrics for mxm id {mxm_id}. Status code {int(lyrics_response.message.header.status_code)}')
        return None
    
    return lyrics_response.message.body.lyrics

def lyrics_response_to_tuple(mxm_id, lyrics_response) -> Tuple[int, int, str, int, str, str]:
    return (
        mxm_id,
        int(lyrics_response.lyrics_id),
        strip_warning(lyrics_response.lyrics_body),
        int(lyrics_response.explicit),
        lyrics_response.lyrics_language,
        lyrics_response.lyrics_language_description
    )

def strip_warning(lyrics_body) -> str:
    # assumes lyrics body strings are normal.
    parts = lyrics_body.split('*******')
    return parts[0]

def insert_lyrics(conn, mxm_id, lyrics_response) -> None:
    insert_lyrics_sql = """
        INSERT INTO lyrics VALUES (?,?,?,?,?,?)
    """
    try:
        cursor = conn.cursor()
        cursor.execute(insert_lyrics_sql, lyrics_response_to_tuple(mxm_id, lyrics_response))
    except SqlException as sqlex:
        print(f'Error inserting lyrics for mxm id {int(lyrics_response.lyrics_id)}.')
        print(sqlex)

if __name__ == '__main__':
    main()