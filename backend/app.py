import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from cryptography.fernet import Fernet
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine, String, Integer, Text, ForeignKey, UniqueConstraint, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.exc import IntegrityError
from policy import decide
from vision import Vision

DATABASE_URL = os.environ["DATABASE_URL"]
ADMIN_KEY = os.environ["ADMIN_KEY"]
if len(ADMIN_KEY) < 24:
    raise RuntimeError("ADMIN_KEY должен содержать минимум 24 символа")
cipher = Fernet(os.environ["EMBEDDING_KEY"].encode())
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
vision = Vision()
lock = threading.Lock()
challenges = {}
attempts = {}
class Base(DeclarativeBase): pass
class Student(Base):
    __tablename__ = "students"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    group: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[str] = mapped_column(Text)
    consent_at: Mapped[str] = mapped_column(String(40))
class Lesson(Base):
    __tablename__ = "lessons"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(120))
    group: Mapped[str] = mapped_column(String(64))
    active: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String(40))
class Attendance(Base):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("student_id", "lesson_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"))
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id"))
    marked_at: Mapped[str] = mapped_column(String(40))
    method: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(String(250), default="")
def now(): return datetime.now(timezone.utc).isoformat()
def auth(x_admin_key: str = Header(default="")):
    if not hmac.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(401, "Неверный ключ преподавателя")
def db():
    with Session(engine) as s: yield s
@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine)
    yield
app = FastAPI(title="Campus Attendance", lifespan=lifespan, dependencies=[Depends(auth)])
class LessonIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    group: str = Field(min_length=1, max_length=64)
class Capture(BaseModel):
    challenge_id: str
    frames: list[str] = Field(min_length=6, max_length=6)
    @field_validator("frames")
    @classmethod
    def bounded_frames(cls, frames):
        if any(len(x)>1_000_000 for x in frames):
            raise ValueError("Кадр слишком большой")
        return frames
class Enroll(Capture):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    group: str = Field(min_length=1, max_length=64)
    consent: bool
class Mark(Capture):
    lesson_id: int
class Manual(BaseModel):
    student_id: int
    lesson_id: int
    reason: str = Field(min_length=5,max_length=250)
class ChallengeIn(BaseModel):
    purpose: str
    lesson_id: int | None = None

def extract(body, purpose, lesson_id=None):
    with lock:
        c = challenges.pop(body.challenge_id, None)
        if not c or c["purpose"] != purpose or c["lesson_id"] != lesson_id or not 3 <= time.monotonic()-c["time"] <= 90:
            raise HTTPException(400,"Проверка истекла или уже использована. Начните снова")
        try: return vision.sequence(body.frames, c["direction"])
        except ValueError as e: raise HTTPException(422,str(e))
        except Exception as e: raise HTTPException(503,"Распознавание недоступно: " + str(e))

@app.get("/api/health")
def health():
    with lock:
        try: vision.load()
        except Exception: pass
    return {"ready": vision.ready, "detail": vision.error, "mode":"local supervised pilot"}
@app.get("/api/students")
def students(s: Session=Depends(db)):
    return [{"id":x.id,"code":x.code,"name":x.name,"group":x.group} for x in s.scalars(select(Student).order_by(Student.name))]
@app.post("/api/challenge")
def challenge(b:ChallengeIn,s:Session=Depends(db)):
    if b.purpose not in ("enroll","verify"): raise HTTPException(400,"Неверная операция")
    if b.purpose == "verify": get_lesson(s,b.lesson_id)
    with lock:
        t=time.monotonic()
        recent=attempts.setdefault("kiosk",[])
        recent[:]=[x for x in recent if t-x<60]
        if len(recent)>=15: raise HTTPException(429,"Слишком много попыток. Подождите минуту")
        recent.append(t)
        for k in list(challenges):
            if t-challenges[k]["time"]>90: del challenges[k]
        token=secrets.token_urlsafe(32); direction=secrets.choice([-1,1])
        challenges[token]={"time":t,"direction":direction,"purpose":b.purpose,"lesson_id":b.lesson_id}
    return {"id":token,"direction":direction,"expires_in":90}
@app.post("/api/students")
def enroll(b:Enroll,s:Session=Depends(db)):
    if not b.consent: raise HTTPException(400,"Требуется согласие и проверка личности сотрудником")
    v=extract(b,"enroll")
    # Reject near-duplicate registrations; administrator must resolve manually.
    for x in s.scalars(select(Student)):
        old=json.loads(cipher.decrypt(x.embedding.encode()))
        if sum(a*z for a,z in zip(v,old)) >= float(os.getenv("MATCH_THRESHOLD","0.55")):
            raise HTTPException(409,"Похожий шаблон уже зарегистрирован. Проверьте существующих студентов")
    x=Student(code=b.code.strip(),name=b.name.strip(),group=b.group.strip(),
              embedding=cipher.encrypt(json.dumps(v.tolist()).encode()).decode(),consent_at=now())
    s.add(x)
    try: s.commit()
    except IntegrityError: s.rollback(); raise HTTPException(409,"Такой номер студента уже существует")
    return {"id":x.id,"message":"Студент зарегистрирован"}
@app.delete("/api/students/{student_id}")
def delete(student_id:int,s:Session=Depends(db)):
    x=s.get(Student,student_id)
    if not x: raise HTTPException(404,"Студент не найден")
    for row in s.scalars(select(Attendance).where(Attendance.student_id==student_id)): s.delete(row)
    s.delete(x); s.commit()
    return {"message":"Студент, шаблон и его отметки удалены"}
@app.get("/api/lessons")
def lessons(s:Session=Depends(db)):
    return [{"id":x.id,"title":x.title,"group":x.group,"active":bool(x.active),"created_at":x.created_at}
            for x in s.scalars(select(Lesson).order_by(Lesson.id.desc()))]
@app.post("/api/lessons")
def create_lesson(b:LessonIn,s:Session=Depends(db)):
    x=Lesson(title=b.title.strip(),group=b.group.strip(),created_at=now());s.add(x);s.commit()
    return {"id":x.id}
@app.post("/api/lessons/{lesson_id}/close")
def close(lesson_id:int,s:Session=Depends(db)):
    x=get_lesson(s,lesson_id);x.active=0;s.commit();return {"message":"Занятие закрыто"}
def get_lesson(s,lesson_id):
    x=s.scalar(select(Lesson).where(Lesson.id==lesson_id).with_for_update())
    if not x or not x.active: raise HTTPException(409,"Занятие не открыто")
    return x

def save_mark(s,student_id,lesson_id,method,reason=""):
    existing=s.scalar(select(Attendance).where(Attendance.student_id==student_id,Attendance.lesson_id==lesson_id))
    if existing: return {"message":"Уже отмечен", "student":s.get(Student,student_id).name}
    s.add(Attendance(student_id=student_id,lesson_id=lesson_id,method=method,reason=reason,marked_at=now()))
    try: s.commit()
    except IntegrityError: s.rollback(); return {"message":"Уже отмечен","student":s.get(Student,student_id).name}
    return {"message":"Присутствие подтверждено","student":s.get(Student,student_id).name}
@app.post("/api/verify")
def verify(b:Mark,s:Session=Depends(db)):
    lesson=get_lesson(s,b.lesson_id)
    v=extract(b,"verify",b.lesson_id)
    scores=[]
    for x in s.scalars(select(Student).where(Student.group==lesson.group)):
        old=json.loads(cipher.decrypt(x.embedding.encode()))
        scores.append((x.id,float(sum(a*z for a,z in zip(v,old)))))
    try: student_id=decide(scores,float(os.getenv("MATCH_THRESHOLD","0.55")),float(os.getenv("MATCH_MARGIN","0.08")))
    except ValueError as e: raise HTTPException(422,str(e))
    s.refresh(lesson)
    if not lesson.active: raise HTTPException(409,"Занятие уже закрыто")
    return save_mark(s,student_id,lesson.id,"face")
@app.post("/api/manual")
def manual(b:Manual,s:Session=Depends(db)):
    lesson=get_lesson(s,b.lesson_id);student=s.get(Student,b.student_id)
    if not student or student.group!=lesson.group: raise HTTPException(400,"Студент не входит в группу")
    return save_mark(s,student.id,lesson.id,"manual",b.reason)
@app.get("/api/attendance/{lesson_id}")
def attendance(lesson_id:int,s:Session=Depends(db)):
    return [{"id":a.id,"student_id":x.id,"name":x.name,"code":x.code,"marked_at":a.marked_at,"method":a.method,"reason":a.reason}
            for a,x in s.execute(select(Attendance,Student).join(Student).where(Attendance.lesson_id==lesson_id).order_by(Attendance.id.desc()))]

class CameraCheck(BaseModel):
    frame: str = Field(min_length=1, max_length=1_000_000)

@app.post("/api/camera-check")
def camera_check(b:CameraCheck):
    with lock:
        try: return vision.diagnose(b.frame)
        except ValueError as e: raise HTTPException(422,str(e))
        except Exception as e: raise HTTPException(503,"Обработка камеры недоступна: "+str(e))
