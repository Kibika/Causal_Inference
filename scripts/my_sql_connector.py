import pandas as pd
import mysql.connector
from mysql.connector import Error
from scripts.encrypt import code


def DBConnect(dbName=None):

    conn = mysql.connector.connect(host='localhost', user='root', password=str(code),
                         database=dbName, buffered=True)
    cur = conn.cursor()
    return conn, cur

def createDB(dbName: str) -> None:

    conn, cur = DBConnect()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {dbName};")
    conn.commit()
    cur.close()

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


create_diagnosis_table = """
CREATE TABLE DiagnosisInformation (
    id INT UNIQUE,
    diagnosis TEXT DEFAULT NULL,
    radius_mean FLOAT DEFAULT NULL,
    texture_mean FLOAT DEFAULT NULL,
    perimeter_mean FLOAT DEFAULT NULL,
    area_mean FLOAT DEFAULT NULL,
    smoothness_mean FLOAT DEFAULT NULL,
    compactness_mean FLOAT DEFAULT NULL,
    concavity_mean FLOAT DEFAULT NULL,
    concave_points_mean FLOAT DEFAULT NULL,
    symmetry_mean FLOAT DEFAULT NULL,
    fractal_dimension_mean FLOAT DEFAULT NULL,
    radius_se FLOAT DEFAULT NULL,
    texture_se FLOAT DEFAULT NULL,
    perimeter_se FLOAT DEFAULT NULL,
    area_se FLOAT DEFAULT NULL,
    smoothness_se FLOAT DEFAULT NULL,
    compactness_se FLOAT DEFAULT NULL,
    concavity_se FLOAT DEFAULT NULL,
    concave_points_se FLOAT DEFAULT NULL,
    symmetry_se FLOAT DEFAULT NULL,
    fractal_dimension_se FLOAT DEFAULT NULL,
    radius_worst FLOAT DEFAULT NULL,
    texture_worst FLOAT DEFAULT NULL,
    perimeter_worst FLOAT DEFAULT NULL,
    area_worst FLOAT DEFAULT NULL,
    smoothness_worst FLOAT DEFAULT NULL,
    compactness_worst FLOAT DEFAULT NULL,
    concavity_worst FLOAT DEFAULT NULL,
    concave_points_worst FLOAT DEFAULT NULL,
    symmetry_worst FLOAT DEFAULT NULL,
    fractal_dimension_worst FLOAT DEFAULT NULL,
    PRIMARY KEY (id)
  );
 """

def insert_to_diagnosis_table(dbName: str, df: pd.DataFrame, table_name: str) -> None:
    conn, cur = DBConnect(dbName)
    for _, row in df.iterrows():
        sqlQuery = f"""INSERT INTO {table_name} (id, diagnosis, radius_mean, texture_mean, perimeter_mean,
                    area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
                    symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                    compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,radius_worst,
                    texture_worst, perimeter_worst, area_worst, smoothness_worst,
                    compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,fractal_dimension_worst)
             VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s);"""
        data = (row[0], row[1], row[2], row[3], (row[4]), (row[5]), row[6], row[7], row[8], row[9], row[10], row[11],
                row[12], row[13], row[14], row[15], row[16], row[17], (row[18]), (row[19]), row[20], row[21], row[22],
                row[23],  row[24], row[25], row[26], row[27],row[28],  row[29], row[30], row[31])

        try:
            # Execute the SQL command
            cur.execute(sqlQuery, data)
            # Commit your changes in the database
            conn.commit()
            print("Data Inserted Successfully")
        except Exception as e:
            conn.rollback()
            print("Error: ", e)
    return

def db_execute_fetch(*args, many=False, tablename='', rdf=True, **kwargs) -> pd.DataFrame:

    connection, cursor1 = DBConnect(**kwargs)
    if many:
        cursor1.executemany(*args)
    else:
        cursor1.execute(*args)

    # get column names
    field_names = [i[0] for i in cursor1.description]

    # get column values
    res = cursor1.fetchall()

    # get row count and show info
    nrow = cursor1.rowcount
    if tablename:
        print(f"{nrow} records fetched from {tablename} table")

    cursor1.close()
    connection.close()

    # return result
    if rdf:
        return pd.DataFrame(res, columns=field_names)
    else:
        return res

if __name__ == "__main__":
    createDB(dbName='Diagnosis')
    execute_query(mysql.connector.connect(host="localhost", user='root',password=str(code), database="Diagnosis"), create_diagnosis_table)

    df = pd.read_csv('../data/data.csv')

    insert_to_diagnosis_table(dbName='Diagnosis', df=df, table_name='DiagnosisInformation')