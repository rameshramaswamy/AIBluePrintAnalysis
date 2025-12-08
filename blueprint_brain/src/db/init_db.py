from blueprint_brain.src.db.session import engine, Base
from blueprint_brain.src.db.models.project import Project, Document
from blueprint_brain.src.db.models.job import Job

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    init_db()