import cv2


def get_direction(face_cx, face_cy, frame_cx, frame_cy, fw, fh):
    """
    Rule-based agent: decides gaze direction based on face position.
    Uses face width/height as dynamic threshold — no iris tracking needed.
    """
    threshold_x = fw * 0.25
    threshold_y = fh * 0.25

    if face_cx < frame_cx - threshold_x:
        return "Left"
    elif face_cx > frame_cx + threshold_x:
        return "Right"
    elif face_cy < frame_cy - threshold_y:
        return "Up"
    elif face_cy > frame_cy + threshold_y:
        return "Down"
    else:
        return "Center"


def draw_text(frame, text, pos=(30, 30), color=(0, 255, 0)):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)