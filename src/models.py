import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

def createTable(tableName:str)->None:
    create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {tableName} (
            id SERIAL PRIMARY KEY,
            contenttype TEXT,
            contentinfo TEXT,
            contentembedding VECTOR(384)
        );
        '''
    cursor.execute(create_table_query)
    conn.commit()
    print(f"Table '{tableName}' created successfully.")

def storeInfo(tableName:str, contenttype:str, contentinfo:str, contentembedding:list)->None:
    insert_query = f'''
        INSERT INTO {tableName} (contenttype, contentinfo, contentembedding)
        VALUES (%s, %s, %s);
        '''
    cursor.execute(insert_query, (contenttype, contentinfo, contentembedding))
    conn.commit()
    print(f"Data inserted into '{tableName}' successfully.")
    
def getStartupContext()->str:
    search_query = f'''
        SELECT contentinfo
        FROM startupintro
        '''
    cursor.execute(search_query)
    results = cursor.fetchall()
    #return as a complete string
    return " ".join([r[0] for r in results])
    

def getContext(queryEmbedding: list[float], tableName: str, top_k: int = 3) -> list[str]:
    # Convert embedding list to Postgres vector string
    vector_str = "[" + ",".join(map(str, queryEmbedding)) + "]"

    # Build SQL query safely for table name
    search_query = f'''
        SELECT contentinfo
        FROM {tableName}
        ORDER BY contentembedding <-> %s
        LIMIT %s;
    '''
    cursor.execute(search_query, (vector_str, top_k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return [r[0] for r in results]

