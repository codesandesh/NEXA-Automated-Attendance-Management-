# train.py
import os
import csv
import pickle
from pathlib import Path

import face_recognition


BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

USERS_CSV = DATA_DIR / "users.csv"
ENCODINGS_PATH = MODEL_DIR / "encodings.pkl"


def load_users():
    """
    Load user info from data/users.csv

    CSV format:
    Id,Name,RFID
    0,biraj,12112
    1,sandesh,22225
    """
    users = {}

    with USERS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)

        # Normalize headers to avoid subtle whitespace issues
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

        required_fields = {"Id", "Name", "RFID"}
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            normalized = {k.strip(): v.strip() for k, v in row.items()}
            user_id = int(normalized["Id"])
            users[user_id] = {
                "id": user_id,
                "name": normalized["Name"],
                "rfid": normalized["RFID"],
            }

    return users


def parse_folder_name(folder_name: str):
    """
    Expect folder name like 'biraj_0', 'sandesh_1'
    Returns: (name, id)
    """
    parts = folder_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Folder '{folder_name}' must be in format '<name>_<id>', "
            "e.g. 'biraj_0'"
        )
    name, id_str = parts
    try:
        user_id = int(id_str)
    except ValueError:
        raise ValueError(f"Folder '{folder_name}' must end with numeric id")

    return name, user_id


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading users from: {USERS_CSV}")
    users = load_users()
    print(f"[INFO] Loaded {len(users)} users from CSV")

    all_encodings = []
    all_metadata = []

    print(f"[INFO] Scanning known faces folder: {KNOWN_FACES_DIR}")
    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        folder_name = person_dir.name
        try:
            folder_name_part, user_id = parse_folder_name(folder_name)
        except ValueError as e:
            print(f"[WARN] {e}. Skipping folder.")
            continue

        if user_id not in users:
            print(
                f"[WARN] User id {user_id} from folder '{folder_name}' "
                f"not found in CSV. Skipping."
            )
            continue

        user_info = users[user_id]
        print(f"[INFO] Processing {folder_name} -> {user_info}")

        # Loop through images
        image_files = [
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if not image_files:
            print(f"[WARN] No images found in {person_dir}")
            continue

        for img_path in image_files:
            print(f"  [IMG] {img_path.name}")
            try:
                image = face_recognition.load_image_file(str(img_path))
                locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, locations)
            except Exception as e:
                print(f"    [ERROR] Failed to process {img_path}: {e}")
                continue

            if len(encodings) == 0:
                print("    [WARN] No face found. Skipping.")
                continue

            if len(encodings) > 1:
                print("    [WARN] Multiple faces found. Using the first one.")

            encoding = encodings[0]

            all_encodings.append(encoding)
            all_metadata.append(
                {
                    "id": user_info["id"],
                    "name": user_info["name"],
                    "rfid": user_info["rfid"],
                }
            )

    if not all_encodings:
        print("[ERROR] No encodings generated. Check your images.")
        return

    data = {
        "encodings": all_encodings,
        "metadata": all_metadata,
    }

    with ENCODINGS_PATH.open("wb") as f:
        pickle.dump(data, f)

    print(
        f"[SUCCESS] Saved {len(all_encodings)} face encodings "
        f"to {ENCODINGS_PATH}"
    )


if __name__ == "__main__":
    train()
