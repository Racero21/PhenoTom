from PySide6.QtSql import QSqlDatabase, QSqlQuery
import shutil
import os

class DatabaseHandler:
    def __init__(self):
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("image_database.db")

        if not self.db.open():
            print("Error: Unable to open database.")

        self.initSchema()

    def initSchema(self):
        query = QSqlQuery()
        # query.exec_("DROP TABLE IF EXISTS images")
        # query.exec_("DROP TABLE IF EXISTS batches")

        query.exec_(
            """
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            )
            """
        )

        query.exec(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_id INTEGER,
                file_path TEXT,
                projected_area REAL,
                extent_x REAL,
                extent_y REAL,
                eccentricity REAL,
                convex_hull_area REAL,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            )
            """ 
        )
    
    def insertBatch(self, batch_name):
        query = QSqlQuery()
        query.prepare("INSERT INTO batches (name) VALUES (:name)")
        query.bindValue(":name", batch_name)

        if not query.exec_():
            print("Error inserting batch into the database:", query.lastError().text())
            # Handle the error accordingly.
        else:
            return query.lastInsertId()

    def getExistingBatches(self):
        query = QSqlQuery()
        query.exec("SELECT id, name FROM batches")

        batches = []
        while query.next():
            batch_id = query.value(0)
            batch_name = query.value(1)
            batches.append((batch_id, batch_name))

        return batches

    def getBatchCount(self):
        query = QSqlQuery()
        query.exec("SELECT COUNT(*) FROM batches")

        if query.next():
            return query.value(0)
        else:
            return 0

    def getBatchParameters(self, batch_id):
        query = QSqlQuery()
        query.prepare("SELECT * FROM images WHERE batch_id = :batch_id")
                      #file_path, projected_area, extent_x, extent_y, eccentricity, convex_hull_area FROM images WHERE batch_id = :batch_id")
        query.bindValue(":batch_id", batch_id)

        print(query.executedQuery())
        if not query.exec():
            print("Error retrieving batch parameters:", query.lastError().text())
            return None
        
        parameters = []
        while query.next():
            result_id = query.value(0)
            result_batch_id = query.value(1)
            file_path = query.value(2)
            projected_area = query.value(3)
            extent_x = query.value(4)
            extent_y = query.value(5)
            eccentricity = query.value(6)
            convex_hull_area = query.value(7)

            parameters.append((result_id, result_batch_id, file_path, projected_area, extent_x, extent_y, eccentricity, convex_hull_area))

        # print(f"ID: {result_id}, Batch ID: {result_batch_id}, File Path: {file_path}, Projected Area: {projected_area}, Extent X: {extent_x}, Extent Y: {extent_y}, Eccentricity: {eccentricity}, Convex Hull Area: {convex_hull_area}")
        # print(f" FROM GET BATCH ----->>>> {parameters}")
        
        return parameters

    def insertImagePath(self, batch_id, file_path, projected_area, extent_x, extent_y, eccentricity, convex_hull_area):
        query = QSqlQuery()
        query.prepare("INSERT INTO images (batch_id, file_path, projected_area, extent_x, extent_y, eccentricity, convex_hull_area) VALUES (:batch_id, :file_path, :projected_area, :extent_x, :extent_y, :eccentricity, :convex_hull_area)")
        query.bindValue(":batch_id", batch_id)
        query.bindValue(":file_path", file_path)
        query.bindValue(":projected_area", projected_area)
        query.bindValue(":extent_x", extent_x)
        query.bindValue(":extent_y", extent_y)
        query.bindValue(":eccentricity", eccentricity)
        query.bindValue(":convex_hull_area", convex_hull_area)

        print(f"projected area: {projected_area} \n, extent_x: {extent_x}\n, extent_y: {extent_y}\n, eccentricity: {eccentricity}\n, convex hull area: {convex_hull_area}\n")
        # self.db.commit()
        print("Executed Query:", query.executedQuery())  # Log the executed query
        if not query.exec_():
            print("Error inserting image into the database:", query.lastError().text())
            # Handle the error accordingly.

        # Additional debug prints
        print(f"Eccentricity value: {projected_area}, Data type: {type(projected_area)}")
        print(f"Eccentricity value: {extent_x}, Data type: {type(extent_x)}")
        print(f"Eccentricity value: {extent_y}, Data type: {type(extent_y)}")
        print(f"Eccentricity value: {eccentricity}, Data type: {type(eccentricity)}")
        print(f"Inserted eccentricity value: {self.getEccentricityForDebugging(batch_id)}")

    def getEccentricityForDebugging(self, batch_id):
    # Use this method for debugging to fetch the eccentricity value from the database
        query = QSqlQuery()
        query.prepare("SELECT eccentricity FROM images WHERE batch_id = :batch_id")
        query.bindValue(":batch_id", batch_id)

        if not query.exec_():
            print("Error fetching eccentricity for debugging:", query.lastError().text())
            return None

        if query.next():
            return query.value(0)
        else:
            return None

    def deleteBatch(self, batch_id, batch_name):
        batch_folder_path = os.path.join("output", f"batch_{batch_name}_{batch_id}")
        self.deleteImagesFromBatch(batch_id)

        try:
            shutil.rmtree(batch_folder_path)
        except Exception as e:
            print(f"Error deleting batch folder: {e}")

        query = QSqlQuery()
        query.prepare("DELETE FROM batches WHERE id = :batch_id")
        query.bindValue(":batch_id", batch_id)

        if not query.exec():
            print("Error deleting batch:", query.lastError().text())

    def deleteImagesFromBatch(self, batch_id):
        query = QSqlQuery()
        query.prepare("DELETE FROM images WHERE batch_id = :batch_id")
        query.bindValue(":batch_id", batch_id)

        if not query.exec():
            print("Error deleting images from the batch:", query.lastError().text())
    
    def getBatchOutputFolderPath(self, batch_id):
        query = QSqlQuery()
        query.prepare("SELECT name FROM batches WHERE id = :batch_id")
        query.bindValue(":batch_id", batch_id)

        if query.exec_() and query.next():
            batch_name = query.value(0)
            return os.path.join("output", f"batch_{batch_name}_{batch_id}")
        else:
            return None