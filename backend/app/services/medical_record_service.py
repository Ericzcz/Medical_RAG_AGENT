import json
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas import MedicalRecordCreate, MedicalRecordStored

from app.db.session import get_db_connection


FIELD_TYPE_MAP = {
    "allergies": "allergy",
    "symptoms": "symptom",
    "medications": "medication",
    "diagnoses": "diagnosis",
    "procedures": "procedure",
    "vitals": "vital",
}


def build_medical_record_items(record: MedicalRecordCreate) -> list[tuple[str, str]]:
    items = []

    for allergy in record.allergies:
        items.append(("allergy", allergy))

    for symptom in record.symptoms:
        items.append(("symptom", symptom))

    for medication in record.medications:
        items.append(("medication", medication))

    for diagnosis in record.diagnoses:
        items.append(("diagnosis", diagnosis))

    for procedure in record.procedures:
        items.append(("procedure", procedure))

    for vital_name, vital_value in record.vitals.items():
        if vital_value:
            items.append(("vital", f"{vital_name}: {vital_value}"))

    return items


async def save_medical_record(
    user_id: str,
    session_id: str | None,
    record: MedicalRecordCreate,
) -> MedicalRecordStored:
    created_at = datetime.now(timezone.utc)
    stored_record = MedicalRecordStored(
        **record.model_dump(),
        record_id=str(uuid4()),
        user_id=user_id,
        source_session_id=session_id,
        created_at=created_at.isoformat(),
    )

    async with get_db_connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO medical_records (
                    record_id,
                    user_id,
                    source_session_id,
                    record_type,
                    visit_date,
                    notes,
                    source_text,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                stored_record.record_id,
                stored_record.user_id,
                stored_record.source_session_id,
                stored_record.record_type,
                stored_record.visit_date,
                stored_record.notes,
                stored_record.source_text,
                created_at,
            )

            for field_type, field_value in build_medical_record_items(record):
                await conn.execute(
                    """
                    INSERT INTO medical_record_items (
                        item_id,
                        record_id,
                        user_id,
                        field_type,
                        field_value,
                        created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    str(uuid4()),
                    stored_record.record_id,
                    stored_record.user_id,
                    field_type,
                    field_value,
                    created_at,
                )
                
    return stored_record

async def get_medical_records(
    user_id: str,
    field: str = "all",
    limit: int = 20,
) -> list[dict]:
    async with get_db_connection() as conn:
        if field == "all":
            rows = await conn.fetch(
                """
                SELECT
                    record_id::text AS record_id,
                    user_id,
                    source_session_id,
                    record_type,
                    visit_date::text AS visit_date,
                    notes,
                    source_text,
                    created_at::text AS created_at
                FROM medical_records
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )

            return [dict(row) for row in rows]

        field_type = FIELD_TYPE_MAP.get(field, field)

        rows = await conn.fetch(
            """
            SELECT
                item_id::text AS item_id,
                record_id::text AS record_id,
                user_id,
                field_type,
                field_value,
                created_at::text AS created_at
            FROM medical_record_items
            WHERE user_id = $1
              AND field_type = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            user_id,
            field_type,
            limit,
        )

        return [dict(row) for row in rows]