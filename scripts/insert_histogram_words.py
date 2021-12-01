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

    create_examples_table(cursor)

    print('getting all src ids...')
    srcs = cursor.execute("""
        SELECT DISTINCT src from similars_src;
    """).fetchall()

    start = time.time()
    for i, src in enumerate(srcs):
        if i % 1000 == 0:
            print(f'took {time.time()-start} seconds to insert {i} histograms')
            con.commit()
        src_id = src[0]
        words = cursor.execute("""
            SELECT words.ROWID, mxm.lyrics.count, mxm.lyrics.is_test
            FROM mxm.words
            JOIN mxm.lyrics
            ON mxm.words.word = mxm.lyrics.word
            WHERE mxm.lyrics.track_id = ?
        """, (src_id,)).fetchall()
        word_strings = []
        for word in words:
            word_strings.append(f'{word[0]-1}:{word[1]}')
        cursor.execute("""
            INSERT OR IGNORE INTO examples
            VALUES (?, ?); 
        """, (src_id, ','.join(word_strings),))

    print('getting all dest ids...')
    srcs = cursor.execute("""
        SELECT DISTINCT dest from similars_src;
    """).fetchall()

    start = time.time()
    for i, src in enumerate(srcs):
        if i % 1000 == 0:
            print(f'took {time.time()-start} seconds to insert {i} histograms')
            con.commit()
        src_id = src[0]
        words = cursor.execute("""
            SELECT words.ROWID, mxm.lyrics.count, mxm.lyrics.is_test
            FROM mxm.words
            JOIN mxm.lyrics
            ON mxm.words.word = mxm.lyrics.word
            WHERE mxm.lyrics.track_id = ?
        """, (src_id,)).fetchall()
        word_strings = []
        for word in words:
            word_strings.append(f'{word[0]-1}:{word[1]}')
        cursor.execute("""
            INSERT OR IGNORE INTO examples
            VALUES (?, ?); 
        """, (src_id, ','.join(word_strings),))

def create_examples_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS examples (
            track_id TEXT PRIMARY KEY,
            histogram TEXT
        )
    """)

if __name__ == '__main__':
    main()
