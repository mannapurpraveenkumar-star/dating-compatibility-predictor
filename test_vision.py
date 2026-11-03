
from pathlib import Path
from app.vision import detect_faces

image_path = input("Enter the full path of a group photo: ").strip().strip('"')

path = Path(image_path)

if not path.is_file():
    print("Image file not found. Check the path and try again.")
else:
    result = detect_faces(path.read_bytes())

    print("\n--- THE DATING GAME: FACE DETECTION ---")
    print(result["message"])
    print("Image size:", result["image_width"], "x", result["image_height"])

    for face in result["faces"]:
        print(face["label"], ":", face["box"])

    print("\nFace detection test completed.")
