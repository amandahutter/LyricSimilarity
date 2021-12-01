import sqlite3
import time

JOINED_DB_PATH = './data_files/mxm_lastfm.db'
MXM_DB_PATH = './data_files/mxm_dataset.db'
LASTFM_DB_PATH = './data_files/lastfm_similars.db'

def main():
    con = sqlite3.connect(JOINED_DB_PATH)
    cursor = con.cursor()

    cursor.execute('ATTACH DATABASE ? AS mxm', (MXM_DB_PATH,))
    cursor.execute('ATTACH DATABASE ? AS lastfm', (LASTFM_DB_PATH,))

    example_id_rows = get_example_id_rows(cursor)

    create_similars_table(cursor)

    print('Inserting similars')
    start = time.time()
    for i, row in enumerate(example_id_rows):
        if i % 1000 == 0:
            print(f'inserted similars for {i} examples in {time.time()-start} seconds')
        id, is_test = row
        insert_similars(cursor, id, is_test)
    end = time.time()
    print(f'Inserted {len(example_id_rows)} rows in {end-start} seconds')
    
    con.commit()

def get_example_id_rows(cursor):
    print('Getting example ids...')
    start = time.time()
    example_id_rows = cursor.execute("""
        SELECT DISTINCT(lyrics.track_id), is_test
        FROM lyrics
        INNER JOIN lastfm.similars_src
        ON lyrics.track_id = lastfm.similars_src.tid;
    """).fetchall()
    end = time.time()
    print(f'Got {len(example_id_rows)} example ids in {end-start} seconds')
    return example_id_rows

def create_similars_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS similars_src (
            src TEXT,
            dest TEXT,
            score REAL,
            is_test INT,
            PRIMARY KEY (src, dest, is_test)
        );
    """)

def insert_similars(cursor, id, is_test):
    similar_row = cursor.execute("""
        SELECT target
        FROM lastfm.similars_src
        WHERE tid = ?;
    """, (id,)).fetchone()
    similars = similar_row[0].split(',')
    inserts = ["BEGIN TRANSACTION;"]
    for i in range(0, len(similars), 2):
        inserts.append(f"INSERT OR IGNORE INTO similars_src values('{id}', '{similars[i]}', {similars[i+1]}, {is_test});")

    inserts.append("COMMIT;")
    query = '\n'.join(inserts)

    cursor.executescript(query)


if __name__ == '__main__':
    main()
