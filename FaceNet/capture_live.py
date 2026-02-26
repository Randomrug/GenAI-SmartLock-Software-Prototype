import cv2
import time

def capture_face_image(save_path="live.jpg", delay=3):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera not accessible")
        return False

    print(f"📷 Camera ON. Capturing image in {delay} seconds...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = int(time.time() - start_time)
        remaining = delay - elapsed

        cv2.putText(
            frame,
            f"Capturing in {max(0, remaining)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Capture", frame)

        if elapsed >= delay:
            cv2.imwrite(save_path, frame)
            print("[OK] Image captured automatically")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return True
# Alias for compatibility
def capture_face(save_path="live.jpg", delay=3):
    """Capture face from camera"""
    return capture_face_image(save_path, delay)