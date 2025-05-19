import os
import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def save_batches(images, labels, batch_size, prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total_samples = len(images)
    num_batches = total_samples // batch_size + (1 if total_samples % batch_size != 0 else 0)

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total_samples)
        batch_data = {
            b'data': images[start:end],
            b'labels': labels[start:end].tolist()
        }
        filename = os.path.join(output_dir, f"mnist_{prefix}_batch_{i+1}")
        try:
            with open(filename, "wb") as f:
                pickle.dump(batch_data, f)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")

def main(batch_size=10000, output_dir="mnist_batches"):
    try:
        print("Downloading MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='auto')
        X, y = mnist["data"], mnist["target"].astype(np.int32)
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        return

    X = X.astype(np.uint8).reshape(-1, 784)

    print("Splitting into training and test sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1/7, random_state=42
        )
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    save_batches(X_train, y_train, batch_size, prefix="train", output_dir=output_dir)

    test_data = {
        b'data': X_test,
        b'labels': y_test.tolist()
    }
    test_path = os.path.join(output_dir, "mnist_test_batch")
    try:
        with open(test_path, "wb") as f:
            pickle.dump(test_data, f)
        print(f"Saved: {test_path}")
    except Exception as e:
        print(f"Error saving {test_path}: {e}")

if __name__ == "__main__":
    main()