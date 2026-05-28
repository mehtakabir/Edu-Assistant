from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config import DATABASE_URL

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id   = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)


class Student(Base):
    __tablename__ = "students"
    id          = Column(Integer, primary_key=True)
    name        = Column(String, nullable=False)
    attendance  = Column(Float,  default=0.0)
    quiz_marks  = Column(Float,  default=0.0)
    quiz_status = Column(String, default="pending")


# Proper FastAPI dependency injection pattern for DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def setup_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            print("Database already set up.")
            return

        db.add_all([
            User(id=1, name="Nakul", role="student"),
            User(id=2, name="Aslam", role="student"),
            User(id=3, name="Kabir", role="teacher"),
        ])

        db.add_all([
            Student(id=1, name="Nakul", attendance=85.5, quiz_marks=8.0, quiz_status="completed"),
            Student(id=2, name="Aslam", attendance=65.0, quiz_marks=7.0, quiz_status="completed"),
        ])

        db.commit()
        print("Database seeded successfully.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_user_role(user_id: int) -> str:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"No user found with id={user_id}")
        return user.role
    finally:
        db.close()