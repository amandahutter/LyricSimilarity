import sqlite3
import argparse
from sqlite3 import Error as SqlException
import os
import time
import swagger_client as MxmApi
from swagger_client.rest import ApiException
from typing import Tuple, Union

# str | Account api key, to be used in every api call
MxmApi.configuration.api_key['apikey'] = os.getenv('MXM_API_KEY')

LYRICS_DB = "./data_files/mxm_lyrics.db"
MXM_TRAIN_FILE = "./data_files/mxm_dataset_train.txt"
MXM_TEST_FILE = "./data_files/mxm_dataset_test.txt"

class UnhandledHttpException(Exception):
    """
    We got some other HTTP code that isn't handled yet.
    """
    def __init__(self, status_code):
        super().__init__(f'Got status code {status_code} from MXM API. Not sure what to do.')

class RateLimitException(Exception):
    """
    We provided api key but still got 401. Probably due to rate limit exceeded.
    """
    def __init__(self):
        super().__init__("Rate limit exceeded")

def main():
    """
    Parses command line arguments and starts the downloader subroutine.
    """
    parser = argparse.ArgumentParser(description="Download lyrics from mxm")
    parser.add_argument('--wait', help="How many seconds to wait between requests in floating point seconds.", default=.1)
    args = parser.parse_args()
    print(f'Using sqlite3 runtime version {sqlite3.sqlite_version}')
    download_and_insert_lyrics(MXM_TRAIN_FILE, wait=args.wait)
    exit(os.EX_OK)

def should_update_download_history_table(download_history_rowid, stop_row, stop_reason) -> bool:
    """
    Decides whether we should update the download history table.
    """
    # If we don't have information about why we stopped, don't update the download history table.
    # Next program run will just start from the last good checkpoint.
    return download_history_rowid is not None and stop_row is not None and stop_reason is not None

def download_and_insert_lyrics(filepath, wait) -> None:
    """
    Sets up a database connection, creates tables if they do not exist, tries to download and insert lyrics,
    and finally tries to record the download history and close the connection.
    """
    conn = None
    download_history_rowid = None
    stop_row = None
    stop_reason = None
    try:
        conn = sqlite3.connect(LYRICS_DB)
        create_download_history_table(conn)
        create_lyrics_table(conn)
        start_row = get_start_row(conn, filepath)
        download_history_rowid = insert_download_history(conn, filepath, int(time.time()), start_row)
        stop_row, stop_reason, not_found_count = read_mxm_file_and_download_lyrics(conn, filepath, wait, start_row, download_history_rowid)
    except SqlException as sqlex:
        print(sqlex)
    except Exception as ex:
        print(ex)
    finally:
        if conn:
            # Only commit if we know why we are stopping. The program makes checkpoints every 1000 rows.
            if should_update_download_history_table(download_history_rowid, stop_row, stop_reason):
                update_download_history(conn, download_history_rowid, stop_row, stop_reason, not_found_count)
                conn.commit()
                conn.close()
            else:
                conn.close()
                exit(os.EX_SOFTWARE)

def should_create_checkpoint(linenum, start_row) -> bool:
    """
    Checks if we should create a checkpoint.
    """
    return (linenum is not 0) and (linenum % 1000 is 0) and (linenum != start_row)

def read_mxm_file_and_download_lyrics(conn, filepath, wait, start_row, download_history_rowid) -> None:
    """
    Iterates through the mxm data file, starting at start row, and tries to download the lyrics for each song row.
    Tries to handle reasons the download failed and return them so they can be recorded in download history.
    """
    not_found_count=0
    stop_reason = None
    with open(filepath) as inputfile:
        for i, line in enumerate(inputfile):

            # Periodically commit db changes
            if should_create_checkpoint(i, start_row):
                print(f'Checkpoint - committing changes at line {i}.')
                update_download_history(conn, download_history_rowid, i, 'good checkpoint', not_found_count)
                conn.commit()

            # skip all rows before start row
            if i < start_row:
                continue
            elif line[0] is '#' or line[0] is '%':
                continue
            else:
                values = line.split(',')
                mxm_id = values[1]
                try:
                    lyrics_response = download_lyrics_throttled(mxm_id, wait)
                except ApiException as api_ex:
                    print("Error sending request to musix match.")
                    stop_reason = str(api_ex)
                except RateLimitException as rlex:
                    print(rlex)
                    stop_reason = i, str(rlex)
                except UnhandledHttpException as unhandled_http_ex:
                    print(unhandled_http_ex)
                    stop_reason = str(unhandled_http_ex)
                except Exception as ex:
                    print(ex)
                    stop_reason = i, str(ex)
                finally:
                    if stop_reason is not None:
                        percent_not_found = not_found_count/(i-start_row)
                        print(f'{percent_not_found}% of lyrics not found')
                        return (i, stop_reason, not_found_count)

                    if lyrics_response is not None:
                        insert_lyrics(conn, mxm_id, lyrics_response)
                    else:
                        not_found_count+=1

        percent_not_found = not_found_count/(i-start_row)
        print(f'{percent_not_found}% of lyrics not found')
        return (i, "Finished iterating over file.", not_found_count)

def get_start_row(conn, filepath) -> int:
    """
        Checks the download_history table to see if the program perviously tried to download lyrics from the mxm API.
        If it finds an entry, it returns the row number it was able to get to in the mxm dataset.
    """
    get_download_history_rowid_sql = """
        SELECT stop_row
        FROM download_history
        WHERE dataset_filepath = ?
        ORDER BY download_date
        LIMIT 1
    """
    try:
        cursor = conn.cursor()
        cursor.execute(get_download_history_rowid_sql, (filepath,))
        last_row = cursor.fetchone()
        if last_row is not None:
            return last_row[0]
        else:
            return 0
    except SqlException as sqlex:
        print("Error getting stop row from previous download")
        print(sqlex)
        exit(os.EX_SOFTWARE)

def insert_download_history(conn, filepath, date, start_row) -> None:
    """
    Inserts a download history record.
    """
    insert_download_history_sql = """
    INSERT INTO download_history
    VALUES (?,?,?,NULL,NULL,NULL)
    """
    try:
        cursor = conn.cursor()
        cursor.execute(insert_download_history_sql, (filepath, date, start_row,))
        return cursor.lastrowid
    except SqlException as sqlex:
        print("Error inserting download history")
        print(sqlex)
        exit(os.EX_SOFTWARE)

def update_download_history(conn, download_history_rowid, stop_row, stop_reason, not_found_count) -> None:
    """
        Updates a download history record with the row the program stopped at and the reason it stopped.
    """
    update_download_history_sql = """
    UPDATE download_history
    SET stop_row = ?,
        stop_reason = ?,
        not_found_count = ?
    WHERE rowid = ?
    """
    try:
        cursor = conn.cursor()
        cursor.execute(update_download_history_sql, (stop_row, stop_reason, not_found_count, download_history_rowid,))
    except SqlException as sqlex:
        print("Error updating download history")
        raise sqlex

def create_download_history_table(conn) -> None:
    """
        This table records a run of this program. It records the range of rows it tried to download.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS download_history (
        dataset_filepath TEXT,
        download_date INTEGER,
        start_row INTEGER,
        stop_row INTEGER,
        stop_reason TEXT,
        not_found_count INTEGER
    )
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
    except SqlException as sqlex:
        print("Error creating download history table.")
        print(sqlex)
        exit(os.EX_SOFTWARE)

def create_lyrics_table(conn) -> None:
    """
    Creates the lyrics table.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS lyrics (
        mxm_id INTEGER PRIMARY KEY,
        lyrics_id INTEGER,
        lyrics TEXT,
        explicit INTEGER,
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
        exit(os.EX_SOFTWARE)

def download_lyrics_throttled(mxm_id, wait) -> Union[dict, None]:
    """
    Waits before running download_lyrics subroutine.
    """
    time.sleep(wait)
    return download_lyrics(mxm_id)

def download_lyrics(mxm_id) -> Union[dict, None]:
    """
    Tries to download lyrics for an mxm id.
    """
    lyrics_api = MxmApi.LyricsApi()
    lyrics_response = lyrics_api.track_lyrics_get_get(mxm_id)
    status_code = int(lyrics_response.message.header.status_code)
    if (status_code is 200):
        return lyrics_response.message.body.lyrics
    elif (status_code == 401 and os.getenv('MXM_API_KEY') is None):
        print("Please export your API key. Copy .env.template to .env, and fill in the API key. Then run source .env")
        exit(os.EX_USAGE)
    elif (status_code == 401 and os.getenv('MXM_API_KEY') is not None):
        # Looks like mxm sends 401 for both no api key, and also rate limit. This case assumes our API key is valid.
        raise RateLimitException
    elif (status_code == 404):
        print(f'Got 404 for {mxm_id}.')
        return None
    elif (status_code != 200 and status_code > 399):
        print(f'Got {status_code} for {mxm_id}. Raising exception.')
        print(lyrics_response)
        raise UnhandledHttpException(status_code)
    else:
        print(f'Got status code {status_code} when downloading {mxm_id}, but can probably continue.')
        return None

def lyrics_response_to_tuple(mxm_id, lyrics_response) -> Tuple[int, int, str, int, str, str]:
    """
    
    """
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
    # We might try to insert the same lyrics twice if we had to start from a checkpoint.
    insert_lyrics_sql = """
    INSERT INTO lyrics VALUES (?,?,?,?,?,?)
    """
    try:
        cursor = conn.cursor()
        cursor.execute(insert_lyrics_sql, lyrics_response_to_tuple(mxm_id, lyrics_response))
    except SqlException as sqlex:
        print(f'Error inserting lyrics for mxm id {int(lyrics_response.lyrics_id)}.')
        raise sqlex

if __name__ == '__main__':
    main()