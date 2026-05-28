from database import SessionLocal, Student


def _build_response(student, field: str) -> dict:
    """
    Build response with only the requested field + name.
    field can be: attendance, quiz_marks, quiz_status, all
    """
    if field == "attendance":
        return {
            "success":    True,
            "name":       student.name,
            "attendance": student.attendance
        }
    elif field == "quiz_marks":
        return {
            "success":   True,
            "name":      student.name,
            "quiz_marks": student.quiz_marks
        }
    elif field == "quiz_status":
        return {
            "success":     True,
            "name":        student.name,
            "quiz_status": student.quiz_status
        }
    else:
        # field == "all" — return everything
        return {
            "success":     True,
            "name":        student.name,
            "attendance":  student.attendance,
            "quiz_marks":  student.quiz_marks,
            "quiz_status": student.quiz_status
        }


def get_student_info(student_id: int, field: str = "all") -> dict:
    """
    Fetch one student by ID.
    field param controls what is returned:
      attendance  → name + attendance only
      quiz_marks  → name + quiz_marks only
      quiz_status → name + quiz_status only
      all         → everything (default)
    """
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            return {"success": False, "message": f"No student found with id={student_id}"}
        return _build_response(student, field)
    finally:
        db.close()


def get_student_by_name(name: str, field: str = "all") -> dict:
    """
    Fetch one student by name — case insensitive.
    field param controls what is returned — same as get_student_info.
    """
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.name.ilike(f"%{name}%")).first()
        if not student:
            return {"success": False, "message": f"No student found with name '{name}'"}
        return _build_response(student, field)
    finally:
        db.close()


def get_all_student_names() -> list:
    """
    Returns list of all student names from DB dynamically.
    Used in graph.py for name matching in queries.
    When new students are added to DB, name matching works automatically.
    No hardcoding needed.
    """
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        return [s.name for s in students]
    finally:
        db.close()


def get_all_students() -> dict:
    """Fetch all students — only teachers should call this"""
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        return {
            "success":  True,
            "students": [
                {
                    "student_id":  s.id,
                    "name":        s.name,
                    "attendance":  s.attendance,
                    "quiz_marks":  s.quiz_marks,
                    "quiz_status": s.quiz_status
                }
                for s in students
            ]
        }
    finally:
        db.close()