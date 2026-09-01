import shutil
import os

db_path = "my-rag-pipeline/my_vector_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Deleted {db_path}")
else:
    print(f"{db_path} does not exist")
