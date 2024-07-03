import face_recognition as fr
import cv2
import numpy as np
import os

# save image in the resources/annotated_faces
# test it out on variety of faces (diverse data)
# clean up the code

faces_dir = "./resources/faces_data"
unknown_faces_dir = "./resources/unknown_faces"


def get_known_face_encodings():
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(faces_dir, filename)
            known_image = fr.load_image_file(image_path)
            encoding = fr.face_encodings(known_image)[0]
            if len(encoding) == 0:
                raise ValueError("No faces found in the known image.")
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0].title())

    # known_image = fr.load_image_file("./resources/faces_data/donald trump.jpg")
    # unknown_image = fr.load_image_file("./resources/unknown_faces/test.jpg")
    # known_face_encodings = fr.face_encodings(known_image)
    # if len(known_face_encodings) == 0:
    #     raise ValueError("No faces found in the known image.")
    # known_face_encoding = known_face_encodings[0]
    # unknown_face_encodings = fr.face_encodings(unknown_image)
    # if len(unknown_face_encodings) == 0:
    #     raise ValueError("No faces found in the unknown image.")
    # for unknown_face_encoding in unknown_face_encodings:
    #     results = fr.compare_faces([known_face_encoding], unknown_face_encoding)
    #     if results[0]:
    #         print("It's a match!")
    #     else:
    #         print("It's not a match.")
    # draw_bounding_boxes(known_image)
    return known_face_encodings, known_face_names


def match_unknown_face(known_face_encodings, known_face_names):
    for filename in os.listdir(unknown_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(unknown_faces_dir, filename)
            unknown_image = fr.load_image_file(image_path)
            unknown_face_encodings = fr.face_encodings(unknown_image)
            unknown_image_rgb = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
            face_locations = fr.face_locations(unknown_image)
            for face_location, unknown_face_encoding in zip(
                face_locations, unknown_face_encodings
            ):
                matches = fr.compare_faces(known_face_encodings, unknown_face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    print("it's a match!", name)

                top, right, bottom, left = face_location
                cv2.rectangle(
                    unknown_image_rgb,
                    (left - 20, top - 20),
                    (right + 20, bottom + 20),
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(
                    unknown_image_rgb,
                    (left - 20, bottom - 15),
                    (right + 20, bottom + 40),
                    (255, 0, 0),
                    cv2.FILLED,
                )
                cv2.putText(
                    unknown_image_rgb,
                    name,
                    (left, bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow("Image", unknown_image_rgb)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_bounding_boxes(image, name):
    unknown_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(unknown_image_rgb, (left, top), (right, bottom), (0, 0, 255), 2)
    # change this to save image in the annotated_faces instead
    cv2.imshow("Image", unknown_image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    known_face_encodings, known_face_names = get_known_face_encodings()
    match_unknown_face(known_face_encodings, known_face_names)


if __name__ == "__main__":
    main()
