from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, LargeBinary

class Base(DeclarativeBase):
    pass

class Face(Base):
    __tablename__ = 'faces'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
