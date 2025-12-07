import threading
import pickle
from pathlib import Path
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import serial

# ----------------- PATHS -----------------
BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_PATH = BASE_DIR / "model" / "encodings.pkl"

# import your train script
import train  # makes train.train() available


# ----------------- FACE LOADING -----------------
def load_known_faces():
    if not ENCODINGS_PATH.exists():
        raise FileNotFoundError(
            f"Encodings file not found: {ENCODINGS_PATH}.\n"
            f"Click 'Train Faces' first."
        )

    with ENCODINGS_PATH.open("rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    metadata = data["metadata"]

    print(f"[INFO] Loaded {len(known_encodings)} known face encodings.")
    return known_encodings, metadata


# ----------------- APP CLASS -----------------
class FaceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NEXA Face Recognition")

        # state
        self.known_encodings = []
        self.metadata = []
        self.camera_running = False
        self.video = None
        self.current_match_id = None
        self.serial_running = False
        self.serial_thread = None
        self.serial_conn = None

        # users / attendance
        self.users = train.load_users()
        self.attendance = {
            user_id: {
                "id": user_id,
                "name": user["name"],
                "rfid": user["rfid"],
                "status": "Absent",
                "time": "-",
            }
            for user_id, user in self.users.items()
        }

        # --- UI variables ---
        self.status_var = tk.StringVar(value="Ready.")
        self.name_var = tk.StringVar(value="-")
        self.rfid_var = tk.StringVar(value="-")
        self.last_seen_var = tk.StringVar(value="-")
        self.serial_port_var = tk.StringVar(value="COM3")

        # --- Layout ---
        self.build_ui()

    def build_ui(self):
        # Top buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=10, pady=10, fill="x")

        train_btn = tk.Button(
            btn_frame, text="Train Faces", command=self.on_train_click, width=15
        )
        train_btn.pack(side="left", padx=5)

        load_btn = tk.Button(
            btn_frame, text="Load Model", command=self.on_load_click, width=15
        )
        load_btn.pack(side="left", padx=5)

        start_btn = tk.Button(
            btn_frame, text="Start Camera", command=self.on_start_camera, width=15
        )
        start_btn.pack(side="left", padx=5)

        stop_btn = tk.Button(
            btn_frame, text="Stop Camera", command=self.on_stop_camera, width=15
        )
        stop_btn.pack(side="left", padx=5)

        # Info frame
        info_frame = tk.LabelFrame(self.root, text="Current Recognition")
        info_frame.pack(padx=10, pady=10, fill="x")

        # Name
        tk.Label(info_frame, text="Name:", width=10, anchor="w").grid(
            row=0, column=0, sticky="w", padx=5, pady=3
        )
        tk.Label(info_frame, textvariable=self.name_var).grid(
            row=0, column=1, sticky="w", padx=5, pady=3
        )

        # RFID
        tk.Label(info_frame, text="RFID:", width=10, anchor="w").grid(
            row=1, column=0, sticky="w", padx=5, pady=3
        )
        tk.Label(info_frame, textvariable=self.rfid_var).grid(
            row=1, column=1, sticky="w", padx=5, pady=3
        )

        # Last seen
        tk.Label(info_frame, text="Last Seen:", width=10, anchor="w").grid(
            row=2, column=0, sticky="w", padx=5, pady=3
        )
        tk.Label(info_frame, textvariable=self.last_seen_var).grid(
            row=2, column=1, sticky="w", padx=5, pady=3
        )

        # RFID verification input
        verify_frame = tk.LabelFrame(self.root, text="RFID Verification")
        verify_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(verify_frame, text="Scan/Enter RFID:", width=15).pack(
            side="left", padx=(5, 2), pady=5
        )
        self.rfid_entry = tk.Entry(verify_frame, width=25)
        self.rfid_entry.pack(side="left", padx=2, pady=5)
        tk.Button(
            verify_frame, text="Verify Attendance", command=self.on_verify_rfid
        ).pack(side="left", padx=6, pady=5)

        # Serial controls
        tk.Label(verify_frame, text="Port:", width=6).pack(
            side="left", padx=(10, 2), pady=5
        )
        tk.Entry(
            verify_frame, textvariable=self.serial_port_var, width=10
        ).pack(side="left", padx=2, pady=5)
        tk.Button(
            verify_frame, text="Connect RFID", command=self.start_serial
        ).pack(side="left", padx=4, pady=5)
        tk.Button(
            verify_frame, text="Disconnect", command=self.stop_serial
        ).pack(side="left", padx=4, pady=5)

        # Attendance table
        table_frame = tk.LabelFrame(self.root, text="Attendance")
        table_frame.pack(padx=10, pady=10, fill="both", expand=True)

        columns = ("Id", "Name", "RFID", "Status", "Time")
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=8
        )
        for col, width in zip(columns, (50, 120, 120, 80, 160)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

        self.tree_items = {}
        self._populate_attendance_table()

        # Status bar
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(status_frame, textvariable=self.status_var, anchor="w").pack(
            fill="x"
        )

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _populate_attendance_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for user_id, record in sorted(self.attendance.items()):
            item_id = self.tree.insert(
                "",
                "end",
                values=(
                    record["id"],
                    record["name"],
                    record["rfid"],
                    record["status"],
                    record["time"],
                ),
            )
            self.tree_items[user_id] = item_id

    def _refresh_attendance_row(self, user_id: int):
        record = self.attendance[user_id]
        item_id = self.tree_items.get(user_id)
        if item_id:
            self.tree.item(
                item_id,
                values=(
                    record["id"],
                    record["name"],
                    record["rfid"],
                    record["status"],
                    record["time"],
                ),
            )

    # ----------------- BUTTON HANDLERS -----------------
    def on_train_click(self):
        self.set_status("Training started...")
        self.root.update_idletasks()

        def _train():
            try:
                train.train()
                self.set_status("Training complete. Loading model...")
                self.known_encodings, self.metadata = load_known_faces()
                self.set_status(
                    f"Model loaded with {len(self.known_encodings)} faces."
                )
            except Exception as e:
                self.set_status(f"Training failed: {e}")
                messagebox.showerror("Error", f"Training failed:\n{e}")

        threading.Thread(target=_train, daemon=True).start()

    def on_load_click(self):
        self.set_status("Loading model...")
        try:
            self.known_encodings, self.metadata = load_known_faces()
            self.set_status(
                f"Model loaded with {len(self.known_encodings)} faces."
            )
        except Exception as e:
            self.set_status(f"Load failed: {e}")
            messagebox.showerror("Error", f"Load failed:\n{e}")

    def on_start_camera(self):
        if self.camera_running:
            self.set_status("Camera already running.")
            return

        if not self.known_encodings:
            try:
                self.known_encodings, self.metadata = load_known_faces()
            except Exception as e:
                self.set_status(f"Load failed: {e}")
                messagebox.showerror("Error", f"Load failed:\n{e}")
                return

        self.set_status("Starting camera...")
        self.camera_running = True

        thread = threading.Thread(target=self.camera_loop, daemon=True)
        thread.start()

    def on_stop_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.set_status("Stopping camera...")
        else:
            self.set_status("Camera is not running.")

    def on_verify_rfid(self):
        scanned_rfid = self.rfid_entry.get().strip()
        if not scanned_rfid:
            self.set_status("Please enter/scan an RFID.")
            return

        if self.current_match_id is None:
            self.set_status("Show your face to the camera first.")
            return

        user_record = self.attendance.get(self.current_match_id)
        if not user_record:
            self.set_status("Matched face not found in user list.")
            return

        if scanned_rfid != user_record["rfid"]:
            self.set_status("RFID does not match the recognized face.")
            messagebox.showerror(
                "Mismatch", "RFID does not match the recognized face."
            )
            return

        self.mark_attendance(self.current_match_id)
        self.rfid_entry.delete(0, tk.END)

    def on_close(self):
        self.camera_running = False
        self.stop_serial()
        if self.video is not None:
            self.video.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def set_status(self, text: str):
        print("[STATUS]", text)
        self.status_var.set(text)

    # ----------------- RFID SERIAL -----------------
    def start_serial(self):
        if self.serial_running:
            self.set_status("RFID serial already connected.")
            return

        port = self.serial_port_var.get().strip()
        if not port:
            self.set_status("Enter a serial port for RFID (e.g., COM3).")
            return

        try:
            self.serial_conn = serial.Serial(port, 9600, timeout=1)
        except Exception as e:
            self.serial_conn = None
            self.set_status(f"Failed to open serial port {port}: {e}")
            messagebox.showerror("Serial Error", f"Cannot open {port}:\n{e}")
            return

        self.serial_running = True

        def _loop():
            try:
                self.serial_loop()
            finally:
                self.serial_running = False
                if self.serial_conn:
                    self.serial_conn.close()
                    self.serial_conn = None
                self.set_status("RFID serial stopped.")

        self.serial_thread = threading.Thread(target=_loop, daemon=True)
        self.serial_thread.start()
        self.set_status(f"RFID serial connected on {port}.")

    def stop_serial(self):
        if self.serial_running:
            self.serial_running = False
            self.set_status("Stopping RFID serial...")
        else:
            self.set_status("RFID serial is not running.")

    def serial_loop(self):
        while self.serial_running and self.serial_conn:
            try:
                line = self.serial_conn.readline().decode("utf-8", "ignore").strip()
            except Exception as e:
                self.set_status(f"Serial read error: {e}")
                break

            if not line:
                continue

            # Expect the Arduino to send just the RFID code per line
            self.handle_rfid_scan(line)

    def handle_rfid_scan(self, scanned_rfid: str):
        self.rfid_entry.delete(0, tk.END)
        self.rfid_entry.insert(0, scanned_rfid)

        if self.current_match_id is None:
            self.set_status("Face not recognized yet. Show face first.")
            return

        user_record = self.attendance.get(self.current_match_id)
        if not user_record:
            self.set_status("Matched face not in user list.")
            return

        if scanned_rfid != user_record["rfid"]:
            self.set_status("RFID does not match recognized face.")
            return

        self.mark_attendance(self.current_match_id)

    def mark_attendance(self, user_id: int):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = self.attendance[user_id]
        record["status"] = "Present"
        record["time"] = now_str
        self._refresh_attendance_row(user_id)
        self.set_status(f"Attendance marked for {record['name']} at {now_str}")

    # ----------------- CAMERA / RECOGNITION -----------------
    def camera_loop(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            self.set_status("Cannot open camera.")
            messagebox.showerror("Error", "Cannot open camera.")
            self.camera_running = False
            return

        self.set_status("Camera started. Press 'q' in video window to stop.")

        THRESHOLD = 0.45

        while self.camera_running:
            ret, frame = self.video.read()
            if not ret:
                self.set_status("Failed to read frame from camera.")
                break

            # Resize for speed
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            label_to_show = "No face"

            if not face_encodings:
                self.current_match_id = None
                self.name_var.set("-")
                self.rfid_var.set("-")

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                if len(distances) == 0:
                    continue

                best_index = np.argmin(distances)
                best_distance = distances[best_index]

                if best_distance < THRESHOLD:
                    person = self.metadata[best_index]
                    name = person["name"]
                    rfid = person["rfid"]
                    self.current_match_id = person["id"]
                    label = f"{name} ({rfid})"

                    # update UI vars (thread-safe enough for simple app)
                    self.name_var.set(name)
                    self.rfid_var.set(rfid)
                    self.last_seen_var.set(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                else:
                    label = "Unknown"
                    self.name_var.set("Unknown")
                    self.rfid_var.set("-")
                    self.current_match_id = None

                label_to_show = label

                # scale coords back up
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                cv2.rectangle(
                    frame, (left, top), (right, bottom), (0, 255, 0), 2
                )
                cv2.rectangle(
                    frame,
                    (left, bottom - 35),
                    (right, bottom),
                    (0, 255, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

            cv2.imshow("Camera - NEXA Face Recognition", frame)

            # allow user to stop via 'q' too
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.camera_running = False
                break

        # cleanup
        if self.video is not None:
            self.video.release()
            self.video = None
        cv2.destroyAllWindows()
        self.set_status("Camera stopped.")


# ----------------- ENTRYPOINT -----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
