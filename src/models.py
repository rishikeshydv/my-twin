import psycopg2

conn = psycopg2.connect(
    dbname="rishi-twin",
    user="rishikeshyadav",
    password="2175",
    host="localhost",
    port="5432"
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