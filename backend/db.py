from __future__ import annotations

import os
import sqlite3
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DB_PATH = os.path.join(DATA_DIR, 'studynest.db')


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS students(
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            program TEXT,
            password TEXT,
            interests TEXT,
            target_role TEXT,
            skill_level TEXT
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS enrollments(
            student_id TEXT,
            term TEXT,
            status TEXT,
            PRIMARY KEY(student_id, term)
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS missing_documents(
            student_id TEXT,
            doc_name TEXT,
            due_date TEXT,
            status TEXT
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS tickets(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            subject TEXT,
            message TEXT,
            status TEXT,
            created_at TEXT
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS appointments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            slot TEXT,
            purpose TEXT,
            created_at TEXT
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS enquiries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            email TEXT,
            subject TEXT,
            message TEXT,
            created_at TEXT
        )
        '''
    )

    c = cur.execute('SELECT COUNT(*) AS c FROM students').fetchone()['c']
    if c == 0:
        students = [
            ('s1001', 'Nithin Krishnan', 'nithin@example.edu', 'Applied Data Science', 'pass1001', 'python,data science,nlp', 'Data Scientist', 'intermediate'),
            ('s1002', 'Asha Menon', 'asha@example.edu', 'Applied Data Science', 'pass1002', 'python,statistics,visualization', 'Data Analyst', 'beginner'),
            ('s1003', 'Rahul Dev', 'rahul@example.edu', 'Applied Data Science', 'pass1003', 'machine learning,python,sql', 'ML Engineer', 'intermediate'),
            ('s1004', 'Lina Bauer', 'lina@example.edu', 'Applied Data Science', 'pass1004', 'nlp,deep learning,python', 'NLP Engineer', 'advanced'),
        ]
        cur.executemany('INSERT INTO students VALUES(?,?,?,?,?,?,?,?)', students)

        enrollments = [
            ('s1001', 'WS25', 'conditional'),
            ('s1002', 'WS25', 'enrolled'),
            ('s1003', 'WS25', 'blocked'),
            ('s1004', 'WS25', 'enrolled'),
        ]
        cur.executemany('INSERT INTO enrollments VALUES(?,?,?)', enrollments)

        missing_documents = [
            ('s1001', 'Health insurance certificate', '2026-10-10', 'missing'),
            ('s1001', 'Fee confirmation', '2026-10-15', 'missing'),
            ('s1003', 'Residence permit page', '2026-10-14', 'missing'),
        ]
        cur.executemany('INSERT INTO missing_documents VALUES(?,?,?,?)', missing_documents)

    conn.commit()
    conn.close()


def list_students() -> list[dict]:
    conn = get_conn()
    rows = conn.execute('SELECT id, name, email, program, interests, target_role, skill_level FROM students ORDER BY id').fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_student(student_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        'SELECT id, name, email, program, interests, target_role, skill_level FROM students WHERE id=?',
        (student_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_login(student_id: str, password: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        'SELECT id, name, email, program, interests, target_role, skill_level FROM students WHERE id=? AND password=?',
        (student_id, password),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_enrollment(student_id: str, term: str) -> dict | None:
    conn = get_conn()
    row = conn.execute('SELECT * FROM enrollments WHERE student_id=? AND term=?', (student_id, term)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_missing_documents(student_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT doc_name, due_date, status FROM missing_documents WHERE student_id=? AND status='missing'",
        (student_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_ticket(student_id: str | None, subject: str, message: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO tickets(student_id, subject, message, status, created_at) VALUES(?,?,?,?,?)',
        (student_id or 'guest', subject, message, 'open', datetime.utcnow().isoformat()),
    )
    tid = cur.lastrowid
    conn.commit()
    conn.close()
    return int(tid)


def create_appointment(student_id: str | None, purpose: str) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    slot = 'Next available slot: Monday 10:00 AM'
    cur.execute(
        'INSERT INTO appointments(student_id, slot, purpose, created_at) VALUES(?,?,?,?)',
        (student_id or 'guest', slot, purpose, datetime.utcnow().isoformat()),
    )
    aid = cur.lastrowid
    conn.commit()
    conn.close()
    return {'id': int(aid), 'slot': slot, 'purpose': purpose}


def create_enquiry(student_id: str | None, name: str, email: str, subject: str, message: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO enquiries(student_id, name, email, subject, message, created_at) VALUES(?,?,?,?,?,?)',
        (student_id, name, email, subject, message, datetime.utcnow().isoformat()),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return int(eid)
