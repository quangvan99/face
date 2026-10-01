import cv2
import numpy as np
import onnxruntime as ort


class FaceDetector:
    INPUT_SIZE = 640
    STRIDES = (8, 16, 32)
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.outputs = [x.name for x in self.session.get_outputs()]

    def detect(self, image):
        size = self.INPUT_SIZE
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        scale = resized.shape[0] / h
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(padded, 1 / 128, (size, size), (127.5,) * 3, swapRB=True)
        outputs = self.session.run(self.outputs, {self.input: blob})

        boxes, points, scores = [], [], []
        for i, stride in enumerate(self.STRIDES):
            score = outputs[i].ravel()
            centers = np.stack(np.mgrid[:size // stride, :size // stride][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            centers = np.repeat(centers, len(score) // len(centers), axis=0)
            bbox = outputs[i + 3].reshape(-1, 4) * stride
            kps = outputs[i + 6].reshape(-1, 5, 2) * stride
            ids = np.flatnonzero(score >= self.SCORE_THRESHOLD)
            if len(ids):
                c, d = centers[ids], bbox[ids]
                boxes.extend(np.column_stack((c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                                              c[:, 0] + d[:, 2], c[:, 1] + d[:, 3])) / scale)
                points.extend((kps[ids] + c[:, None, :]) / scale)
                scores.extend(score[ids])

        rects = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]
        ids = cv2.dnn.NMSBoxes(rects, list(scores), self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
        return [(boxes[i], points[i]) for i in np.asarray(ids).reshape(-1)] if len(ids) else []


class FaceAlign:
    # FOR InsightFace EMB
    OUTPUT_SIZE = 112
    LANDMARK_TEMPLATE = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041],
    ], dtype=np.float32)

    def align(self, image, bbox, kps):
        # SimilarityTransform least-squares fit used by InsightFace estimate_norm().
        source = kps.astype(np.float64)
        target = self.LANDMARK_TEMPLATE.astype(np.float64)
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        source_centered, target_centered = source - source_mean, target - target_mean
        denominator = np.sum(source_centered ** 2)
        a = np.sum(source_centered[:, 0] * target_centered[:, 0]
                   + source_centered[:, 1] * target_centered[:, 1]) / denominator
        b = np.sum(source_centered[:, 0] * target_centered[:, 1]
                   - source_centered[:, 1] * target_centered[:, 0]) / denominator
        linear = np.array([[a, -b], [b, a]])
        translation = target_mean - linear @ source_mean
        matrix = np.column_stack((linear, translation))
        return cv2.warpAffine(
            image, matrix, (self.OUTPUT_SIZE, self.OUTPUT_SIZE), borderValue=0.0
        )


class FaceEmbedder:
    INPUT_SIZE = 112
    PIXEL_MEAN = 127.5
    PIXEL_STD = 127.5

    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, face_crop):
        if face_crop.shape[:2] != (self.INPUT_SIZE, self.INPUT_SIZE):
            face_crop = cv2.resize(face_crop, (self.INPUT_SIZE, self.INPUT_SIZE))
        blob = cv2.dnn.blobFromImage(
            face_crop, 1 / self.PIXEL_STD, (self.INPUT_SIZE, self.INPUT_SIZE),
            (self.PIXEL_MEAN,) * 3, swapRB=True,
        )
        vector = self.session.run([self.output_name], {self.input_name: blob})[0].flatten()
        return vector / max(np.linalg.norm(vector), 1e-6)


class FaceAnti:
    INPUT_SIZE = 80
    V2_SCALE = 2.7
    V1SE_SCALE = 4.0
    REAL_THRESHOLD = 0.5

    def __init__(self, model_v2_path, model_v1se_path):
        self.models = []
        for path in (model_v2_path, model_v1se_path):
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.models.append((session, session.get_inputs()[0].name,
                                session.get_outputs()[0].name))

    @staticmethod
    def crop(image, bbox, scale):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        scale = min(scale, (h - 1) / max(bh, 1), (w - 1) / max(bw, 1))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, top = max(0, int(cx - bw * scale / 2)), max(0, int(cy - bh * scale / 2))
        right = min(w - 1, int(cx + bw * scale / 2))
        bottom = min(h - 1, int(cy + bh * scale / 2))
        crop = image[top:bottom + 1, left:right + 1]
        return cv2.resize(crop, (FaceAnti.INPUT_SIZE, FaceAnti.INPUT_SIZE))

    def _predict(self, model, face_crop):
        session, input_name, output_name = model
        face_crop = cv2.resize(face_crop, (self.INPUT_SIZE, self.INPUT_SIZE))
        tensor = face_crop.astype(np.float32).transpose(2, 0, 1)[None]
        logits = session.run([output_name], {input_name: tensor})[0][0]
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def check(self, crop_v2, crop_v1se):
        probs = 0.75 * self._predict(self.models[0], crop_v2)
        probs += 0.25 * self._predict(self.models[1], crop_v1se)
        real_score = float(probs[1])
        return {
            "is_real": real_score >= self.REAL_THRESHOLD,
            "real_score": real_score,
            "spoof_score": 1.0 - real_score,
        }
