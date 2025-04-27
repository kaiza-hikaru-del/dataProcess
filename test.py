from dataProcess import process_sample, process_field, process_category

from pathlib import Path
import time

if __name__ == "__main__":
    start = time.time()
    categories = list(Path("E:/Datasets/testDataset").glob("[!.]*"))
    print(categories)
    for cat_dir in categories:
        process_category(cat_dir)
    end = time.time()
    print(f"Time cost: {end - start:.3f}s")
