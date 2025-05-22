import pickle

with open("dist/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Label Index → Phoneme Mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")


