# Test fixtures

| File | What it is | Used to test |
|---|---|---|
| `single_face.jpg` | NASA portrait of astronaut Eileen Collins (public domain), as bundled with scikit-image, resized to 384 px | one face |
| `group_four.jpg/.png/.webp` | Synthetic "group" made by placing four copies of that portrait at different sizes, two mirrored, on a plain background | multiple faces, left-to-right order, all three formats |
| `no_face.jpg` | "Chelsea" the cat, from scikit-image (public domain) | no faces detected |

The group image is a composite, so all four "people" are the same person. That
is fine for testing detection and pairing mechanics, and is a reminder that the
app never identifies anyone. Use real group photos (with consent) for manual testing.
