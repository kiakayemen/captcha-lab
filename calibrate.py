from pathlib import Path

import cv2


IMAGE_PATH = Path("dataset/Screenshot-00023.png")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

display = image.copy()
points: list[tuple[int, int]] = []


def mouse_callback(event, x, y, flags, param):
    global display

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((x, y))

    print(f"Clicked: x={x}, y={y}")

    cv2.circle(
        display,
        (x, y),
        5,
        (0, 0, 255),
        -1,
    )

    if len(points) == 1:
        print("Now click the bottom-right corner of the grid.")

    elif len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Ensure coordinates are ordered correctly.
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        cv2.rectangle(
            display,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2,
        )

        print("\nPut these values in benchmark.py:\n")
        print(f"Y1, Y2 = {top}, {bottom}")
        print(f"X1, X2 = {left}, {right}")
        print()
        print("Equivalent NumPy crop:")
        print(f"grid = image[{top}:{bottom}, {left}:{right}]")

    cv2.imshow("Calibration", display)


cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.imshow("Calibration", display)
cv2.setMouseCallback("Calibration", mouse_callback)

print("Click the top-left corner of the grid.")
print("Press R to reset.")
print("Press Q or Escape to quit.")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key in (ord("q"), 27):
        break

    if key == ord("r"):
        points.clear()
        display = image.copy()
        cv2.imshow("Calibration", display)
        print("\nReset. Click the top-left corner of the grid.")

cv2.destroyAllWindows()
