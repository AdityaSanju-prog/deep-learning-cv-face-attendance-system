"""
Database Management Layer for DeepFaceAttend
============================================
Handles student biometric registration and daily attendance logging using SQLite.
Includes duplicate check-in suppression, query filters, and CSV report export.
"""

import sqlite3
import os
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class AttendanceDatabase:
    """SQLite Database manager for face biometrics and attendance records."""

    def __init__(self, db_path: str = "data/attendance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema with indexing for fast lookups."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Students Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Attendance Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                liveness_score REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'Present',
                FOREIGN KEY (roll_number) REFERENCES students(roll_number),
                UNIQUE(roll_number, date)
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stud_roll ON students(roll_number);")
            conn.commit()

    def register_student(self, roll_number: str, name: str, department: str,
                         embedding: np.ndarray, photo_path: str = "") -> Tuple[bool, str]:
        """
        Enrolls a new student with their computed 512-D normalized facial embedding.
        """
        roll_number = roll_number.strip().upper()
        name = name.strip()
        department = department.strip()

        # Convert numpy array to JSON for clean serialization
        emb_list = embedding.flatten().tolist()
        emb_json = json.dumps(emb_list)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO students (roll_number, name, department, embedding_json, photo_path)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(roll_number) DO UPDATE SET
                    name=excluded.name,
                    department=excluded.department,
                    embedding_json=excluded.embedding_json,
                    photo_path=excluded.photo_path;
                """, (roll_number, name, department, emb_json, photo_path))
                conn.commit()
            return True, f"Student {name} ({roll_number}) enrolled successfully."
        except Exception as e:
            return False, f"Registration failed: {str(e)}"

    def get_all_students(self) -> List[Dict[str, Any]]:
        """Retrieves list of all enrolled students."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, roll_number, name, department, photo_path, created_at FROM students ORDER BY roll_number ASC;")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_gallery_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Loads all enrolled student embeddings into memory for fast vector matching.
        Returns:
            dict mapping roll_number -> 1D numpy array
        """
        gallery = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT roll_number, embedding_json FROM students;")
            rows = cursor.fetchall()
            for row in rows:
                roll = row["roll_number"]
                emb = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
                gallery[roll] = emb
        return gallery

    def get_student_by_roll(self, roll_number: str) -> Optional[Dict[str, Any]]:
        """Looks up student record by Roll Number."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE roll_number = ?;", (roll_number.strip().upper(),))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res["embedding"] = np.array(json.loads(res["embedding_json"]), dtype=np.float32)
                return res
        return None

    def mark_attendance(self, roll_number: str, confidence: float, liveness_score: float) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Marks attendance for the current date.
        If already marked today, prevents duplicate check-in while returning existing record.
        """
        student = self.get_student_by_roll(roll_number)
        if not student:
            return False, f"Roll number {roll_number} not found in student directory.", {}

        today_str = date.today().isoformat()
        time_str = datetime.now().strftime("%H:%M:%S")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check existing check-in today
            cursor.execute("SELECT * FROM attendance WHERE roll_number = ? AND date = ?;", (student["roll_number"], today_str))
            existing = cursor.fetchone()

            if existing:
                record = dict(existing)
                return False, f"Attendance already marked for {student['name']} today at {record['timestamp']}.", record

            cursor.execute("""
            INSERT INTO attendance (roll_number, name, department, date, timestamp, confidence, liveness_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Present');
            """, (student["roll_number"], student["name"], student["department"], today_str, time_str, float(confidence), float(liveness_score)))
            conn.commit()

            new_record = {
                "roll_number": student["roll_number"],
                "name": student["name"],
                "department": student["department"],
                "date": today_str,
                "timestamp": time_str,
                "confidence": round(float(confidence), 4),
                "liveness_score": round(float(liveness_score), 4),
                "status": "Present"
            }
            return True, f"Attendance recorded for {student['name']} ({student['roll_number']}) at {time_str}.", new_record

    def get_attendance_by_date(self, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves attendance entries for a specified date (YYYY-MM-DD), default today."""
        if target_date is None:
            target_date = date.today().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT id, roll_number, name, department, date, timestamp, confidence, liveness_score, status
            FROM attendance
            WHERE date = ?
            ORDER BY timestamp DESC;
            """, (target_date,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_attendance_stats(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Calculates daily attendance summary metrics."""
        if target_date is None:
            target_date = date.today().isoformat()

        all_students = self.get_all_students()
        total_students = len(all_students)
        records = self.get_attendance_by_date(target_date)
        present_count = len(records)
        absent_count = max(0, total_students - present_count)
        rate = (present_count / total_students * 100.0) if total_students > 0 else 0.0

        return {
            "date": target_date,
            "total_students": total_students,
            "present_count": present_count,
            "absent_count": absent_count,
            "attendance_rate": round(rate, 1)
        }

    def export_to_csv(self, target_date: Optional[str] = None, output_path: Optional[str] = None) -> str:
        """Exports the day's attendance records to CSV."""
        if target_date is None:
            target_date = date.today().isoformat()
        if output_path is None:
            output_path = f"data/attendance_records/attendance_{target_date}.csv"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        records = self.get_attendance_by_date(target_date)
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=["id", "roll_number", "name", "department", "date", "timestamp", "confidence", "liveness_score", "status"])
        df.to_csv(output_path, index=False)
        return output_path


if __name__ == "__main__":
    print("[Database Test]")
    db = AttendanceDatabase("data/test_attendance.db")
    dummy_emb = np.random.randn(512)
    dummy_emb /= np.linalg.norm(dummy_emb)

    ok, msg = db.register_student("CS101", "Alice Smith", "Computer Science", dummy_emb)
    print("Enroll:", msg)

    gallery = db.get_gallery_embeddings()
    print("Gallery size:", len(gallery))

    ok2, msg2, rec = db.mark_attendance("CS101", 0.92, 0.88)
    print("Mark 1:", msg2)

    ok3, msg3, rec3 = db.mark_attendance("CS101", 0.95, 0.89)
    print("Mark 2 (Duplicate check):", msg3)

    stats = db.get_attendance_stats()
    print("Stats:", stats)

    # Cleanup test db
    if os.path.exists("data/test_attendance.db"):
        os.remove("data/test_attendance.db")
    print("Database test passed!")
