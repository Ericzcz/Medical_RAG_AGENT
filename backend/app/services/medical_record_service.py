import json
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas import MedicalRecordCreate, MedicalRecordStored


def make_medical_records_key(user_id: str) -> str:
    return f"user:{user_id}:medical_records"


async def save_medical_record(
    redis_client,
    user_id: str,
    session_id: str | None,
    record: MedicalRecordCreate,
) -> MedicalRecordStored:
    stored_record = MedicalRecordStored(
        **record.model_dump(),
        record_id=str(uuid4()),
        user_id=user_id,
        source_session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    key = make_medical_records_key(user_id)

    await redis_client.rpush(
        key,
        json.dumps(stored_record.model_dump(), ensure_ascii=False),
    )

    return stored_record