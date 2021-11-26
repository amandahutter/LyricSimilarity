import sqlite3

class SongNotFoundException(Exception):
    def __init__(self, mxm_id):
        super().__init__(f'Song with mxm id {mxm_id} not found in adjacency list')

class AdjacencyList:
    def __init__(self, similarity_database: str, lyrics_db: str, filter_by_lyrics_db: str):
        self.__adjacency_list = {}
        con = sqlite3.connect(similarity_database)
        cur = con.cursor()

        if filter_by_lyrics_db:
            cur.execute('ATTACH DATABASE ? AS lyrics', (lyrics_db,))

        if filter_by_lyrics_db:
            cur.execute("""
                SELECT *
                FROM similars_src
                WHERE tid in
                    (SELECT DISTINCT track_id FROM lyrics)
                """)
        else:
            cur.execute('SELECT * FROM similars_src')
        rows = cur.fetchall()
        for i, row in enumerate(rows):
            if i % 1000 == 999:
                # Writes these logs on one line
                print('\r' + f'Loaded {i+1} song similarities', end="")
            key = row[0]
            adjacencies = {}
            tokens = row[1].split(',')
            for j in range(0, len(tokens), 2):
                adjacencies[tokens[j]] = float(tokens[j+1])
            self.__adjacency_list[key] = adjacencies
        print('\r' + f'Loaded {i+1} song similarities')

        cur.close()
    
    def get_similarity(self, src, dest):
        if src not in self.__adjacency_list:
            raise SongNotFoundException(src)
        adjacencies = self.__adjacency_list[src]
        if dest in adjacencies:
            return adjacencies[dest]
        else:
            return 0

            
            
            