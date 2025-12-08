from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from blueprint_brain.config.settings import settings

# Create Engine
# pool_pre_ping=True handles DB connection drops gracefully
engine = create_engine(
    settings.DATABASE_URL, 
    pool_pre_ping=True, 
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()