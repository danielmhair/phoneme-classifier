import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
import json

def fine_tune_classifier_on_noisy(
    NOISY_EMBEDDINGS_DIR: Path,
    CLEAN_MODEL_PATH: Path,
    LABEL_ENCODER_PATH: Path,
    FINETUNED_MODEL_PATH: Path,
    FINE_TUNE_EPOCHS = 2,
    FINE_TUNE_LR = 1e-4,
):
    NOISY_METADATA_CSV = NOISY_EMBEDDINGS_DIR / "metadata.csv"
 
    # --- LOAD CLEAN MODEL ---
    with open(CLEAN_MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    # --- CHECK CLASSIFIER OUTPUT SHAPE ---
    expected_classes = 37

    # Try to infer hidden_size and output shape from the classifier
    if hasattr(clf, 'coefs_') and len(clf.coefs_) > 0:
        hidden_size = clf.coefs_[-1].shape[0]
        output_size = clf.coefs_[-1].shape[1]
        print(f"[DEBUG] Classifier output layer shape: ({hidden_size}, {output_size})")
        if output_size != expected_classes:
            raise ValueError(f"Classifier output layer shape mismatch: expected output size {expected_classes}, got {output_size}. Aborting fine-tuning.")
    else:
        print("[WARNING] Could not determine classifier output shape. Skipping shape check.")

    # --- LOAD NOISY EMBEDDINGS ---
    def load_noisy_embeddings():
        import pandas as pd
        meta = pd.read_csv(NOISY_METADATA_CSV)
        print(f"Loaded {len(meta)} rows from {NOISY_METADATA_CSV}")
        X, y = [], []
        missing = 0
        for _, row in meta.iterrows():
            emb_path = NOISY_EMBEDDINGS_DIR / row['embedding_filename']
            if not emb_path.exists():
                print(f"Missing embedding: {emb_path}")
                missing += 1
                continue
            X.append(np.load(emb_path))
            y.append(row['phoneme'])
        print(f"Found {len(X)} embeddings, {missing} missing.")
        return X, y

    X_noisy, y_noisy = load_noisy_embeddings()
    if not X_noisy:
        raise ValueError(f"No embeddings found in {NOISY_EMBEDDINGS_DIR}. Check your metadata and embedding files.")
    X_noisy = np.stack(X_noisy)
    y_noisy = label_encoder.transform(y_noisy)

    # --- CHECK LABEL ENCODER CLASSES MATCH ---
    # Convert np.int64 to int for readable error messages
    noisy_classes = set(int(x) for x in np.unique(y_noisy))
    model_classes = set(int(x) for x in getattr(clf, 'classes_', []))
    if model_classes and noisy_classes != model_classes:
        missing_in_noisy = model_classes - noisy_classes
        missing_in_model = noisy_classes - model_classes
        raise ValueError(f"Label mismatch for warm_start: Model classes: {sorted(model_classes)}, Noisy y classes: {sorted(noisy_classes)}.\nMissing in noisy: {sorted(missing_in_noisy)}\nMissing in model: {sorted(missing_in_model)}\nAborting fine-tune to prevent sklearn warm_start error.")

    # --- EVALUATE BEFORE FINE-TUNING ---
    y_pred_before = clf.predict(X_noisy)
    acc_before = accuracy_score(y_noisy, y_pred_before)
    print(f"[Before fine-tune] Accuracy on noisy set: {acc_before:.2%}")

    # --- FINE-TUNE ---
    # Check output layer shape matches number of classes
    n_classes = len(label_encoder.classes_)
    output_shape = clf.coefs_[-1].shape[1]
    if output_shape != n_classes:
        raise ValueError(f"Classifier output shape mismatch: got {output_shape} outputs, but label encoder has {n_classes} classes. Aborting fine-tune to prevent ONNX/class mismatch.")
    print(f"[Shape check] Classifier output: {output_shape}, Label encoder classes: {n_classes}")
    print(f"Fine-tuning for {FINE_TUNE_EPOCHS} epochs with lr={FINE_TUNE_LR}...")
    clf.set_params(max_iter=FINE_TUNE_EPOCHS, learning_rate_init=FINE_TUNE_LR, warm_start=True)
    clf.fit(X_noisy, y_noisy)

    # --- EVALUATE AFTER FINE-TUNING ---
    y_pred_after = clf.predict(X_noisy)
    acc_after = accuracy_score(y_noisy, y_pred_after)
    print(f"[After fine-tune] Accuracy on noisy set: {acc_after:.2%}")
    print(classification_report(y_noisy, y_pred_after, target_names=label_encoder.classes_))

    # --- CHECK PHONEME LABELS JSON CONSISTENCY ---
    PHONEME_LABELS_JSON_PATH = NOISY_EMBEDDINGS_DIR.parent / "phoneme_labels.json"
    if PHONEME_LABELS_JSON_PATH.exists():
        with open(PHONEME_LABELS_JSON_PATH, "r", encoding="utf-8") as f:
            phoneme_labels = json.load(f)
        n_labels_json = len(phoneme_labels)
        if n_labels_json != n_classes:
            raise ValueError(f"phoneme_labels.json has {n_labels_json} labels, but label encoder has {n_classes} classes. Aborting fine-tune to prevent class mismatch.")
        print(f"[Shape check] phoneme_labels.json count: {n_labels_json}, matches label encoder classes: {n_classes}")
    else:
        print(f"[WARNING] phoneme_labels.json not found at {PHONEME_LABELS_JSON_PATH}. Skipping label count check.")

    # --- SAVE FINETUNED MODEL ---
    with open(FINETUNED_MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"✅ Fine-tuned model saved to {FINETUNED_MODEL_PATH}")
