import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create() if hasattr(cv2.face, 'LBPHFaceRecognizer_create') else None
if recognizer is None:
    messagebox.showerror("Error", "OpenCV 'face' module is missing. Install 'opencv-contrib-python'")
    exit()

model_file = "face_trained.yml"
label_file = "label_dict.npy"

if os.path.exists(model_file) and os.path.exists(label_file):
    recognizer.read(model_file)
    label_dict = np.load(label_file, allow_pickle=True).item()
else:
    label_dict = {}

def capture_face():
    name = simpledialog.askstring("Input", "Enter your name:")
    if not name:
        return

    user_folder = os.path.join(dataset_path, name)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(f"{user_folder}/{count}.jpg", face)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Captured {count} images for {name}!")

def train_model():
    faces, labels = [], []
    label_dict.clear()
    label_count = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            label_dict[label_count] = person_name
            for image_name in os.listdir(person_path):
                img_path = os.path.join(person_path, image_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                labels.append(label_count)
            label_count += 1

    if faces and labels:
        recognizer.train(faces, np.array(labels))
        recognizer.save(model_file)
        np.save(label_file, label_dict)
        messagebox.showinfo("Success", "Model trained successfully!")
    else:
        messagebox.showerror("Error", "No face data found, Please capture faces first.")

def recognize_face():
    if not os.path.exists(model_file) or not os.path.exists(label_file):
        messagebox.showerror("Error", "No trained model found, train the model first.")
        return

    recognizer.read(model_file)
    label_dict = np.load(label_file, allow_pickle=True).item()

    cap = cv2.VideoCapture(0)

    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", screen_width // 2, screen_height // 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face_resized)
            name = "Unknown"

            if confidence < 80:
                name = label_dict.get(label, "Unknown")

            cv2.putText(frame, f"{name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 153, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 153, 255), 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

def manage_users():
    users = os.listdir(dataset_path)

    if not users:
        messagebox.showinfo("Info", "No user data available!")
        return

    manage_window = tk.Toplevel(root)
    manage_window.title("Manage Users")
    manage_window.geometry("300x400")

    tk.Label(manage_window, text="Stored Users:", font=("Arial", 12, "bold")).pack(pady=10)

    listbox = tk.Listbox(manage_window)
    listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    for user in users:
        listbox.insert(tk.END, user)

    def edit_user():
        selected_index = listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Please select a user to rename!")
            return

        old_name = listbox.get(selected_index)
        new_name = simpledialog.askstring("Rename User", f"Enter new name for '{old_name}':")

        if new_name and new_name != old_name:
            old_path = os.path.join(dataset_path, old_name)
            new_path = os.path.join(dataset_path, new_name)

            os.rename(old_path, new_path)

            listbox.delete(selected_index)
            listbox.insert(selected_index, new_name)
            messagebox.showinfo("Success", f"Renamed '{old_name}' to '{new_name}'")

            retrain_model()

    def delete_user():
        selected_index = listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Please select a user to delete!")
            return

        user = listbox.get(selected_index)
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{user}'?")

        if confirm:
            user_path = os.path.join(dataset_path, user)
            for file in os.listdir(user_path):
                os.remove(os.path.join(user_path, file))
            os.rmdir(user_path)
            listbox.delete(selected_index)
            messagebox.showinfo("Success", f"Deleted user '{user}'")

            retrain_model()

    def retrain_model():
        faces, labels = [], []
        new_label_dict = {}
        new_label_count = 0

        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_path):
                new_label_dict[new_label_count] = person_name
                for image_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, image_name)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        faces.append(image)
                        labels.append(new_label_count)
                new_label_count += 1

        if faces and labels:
            recognizer.train(faces, np.array(labels))
            recognizer.save(model_file)
            np.save(label_file, new_label_dict)
            messagebox.showinfo("Success", "Model retrained after user rename!")
        else:

            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(label_file):
                os.remove(label_file)
            messagebox.showinfo("Info", "All users deleted. Model reset.")

    tk.Button(manage_window, text="Edit Name", command=edit_user).pack(pady=5)
    tk.Button(manage_window, text="Delete User", command=delete_user).pack(pady=5)

root = tk.Tk()
root.title("Face Recognition System")
root.geometry("300x120")

tk.Button(root, text="Capture Face Data", command=capture_face).pack()
tk.Button(root, text="Train Model", command=train_model).pack()
tk.Button(root, text="Face Recognition", command=recognize_face).pack()
tk.Button(root, text="Manage Users", command=manage_users).pack()

root.mainloop()