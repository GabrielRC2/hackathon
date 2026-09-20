import faiss
import numpy as np


class VectorDatabase:

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.ids = []

    def add(self, object_id, vector):

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        vector = vector.reshape(
            1,
            -1
        )

        self.index.add(vector)

        self.ids.append(object_id)

    def search(self, vector, k=5):

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        vector = vector.reshape(
            1,
            -1
        )

        scores, indices = self.index.search(
            vector,
            k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "id": self.ids[index],
                "score": float(score)
            })

        return results