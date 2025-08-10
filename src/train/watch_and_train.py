import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import os

DATA_PATH = os.path.abspath("data/california_housing.csv")  # path to your dataset

class DatasetChangeHandler(FileSystemEventHandler):
    def __init__(self, file_to_watch):
        super().__init__()
        self.file_to_watch = file_to_watch

    def on_modified(self, event):
        try:
            if os.path.exists(event.src_path) and os.path.samefile(event.src_path, self.file_to_watch):
                print(f"[Watcher] Detected change in dataset: {event.src_path}")
                threading.Thread(target=self.run_training, daemon=True).start()
        except FileNotFoundError:
            pass  # This can happen if the file is replaced very quickly

    def run_training(self):
        print("[Watcher] Starting retraining job in background...")
        try:
            # Train linear regression
            subprocess.run(
                [
                    "python",
                    "-m", "src.train.train_models",
                    "--data-path", DATA_PATH,
                    "--algorithms", "linear_regression"
                ],
                check=True
            )

            # Train decision tree
            subprocess.run(
                [
                    "python",
                    "-m", "src.train.train_models",
                    "--data-path", DATA_PATH,
                    "--algorithms", "decision_tree"
                ],
                check=True
            )

            # Select best model
            subprocess.run(
                [
                    "python",
                    "-m", "src.models.best_model_selector"
                ],
                check=True
            )

            print("[Watcher] ✅ Retraining completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[Watcher] ❌ Error during training: {e}")

if __name__ == "__main__":
    event_handler = DatasetChangeHandler(DATA_PATH)
    observer = Observer()
    watch_dir = os.path.dirname(DATA_PATH)
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()

    print(f"[Watcher] Monitoring {DATA_PATH} for changes... Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
