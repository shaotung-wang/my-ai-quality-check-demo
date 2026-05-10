"""
app.core.pipeline_demo

Offline demo: simulate multiple cameras feeding images and run inference+aggregation.
"""
import time
import os
import glob
import cv2
from app.core.inference import ModelInfer
from app.core.aggregator import Aggregator
from app.core import config as core_config


def load_sample_images(limit_per_cam=20):
	img_glob = os.path.join('datasets', 'metal_defects', 'train', 'images', '*.jpg')
	all_imgs = sorted(glob.glob(img_glob))
	if not all_imgs:
		print('No sample images found under datasets/metal_defects/train/images')
		return [[] for _ in core_config.CAMERA_LIST]

	cams = [[] for _ in core_config.CAMERA_LIST]
	for i, p in enumerate(all_imgs):
		cams[i % len(cams)].append(p)
		if all(len(c) >= limit_per_cam for c in cams):
			break

	return cams


def simulate_run(num_shafts=10, frames_per_shaft=1):
	cams_images = load_sample_images(limit_per_cam=100)
	infer = ModelInfer()
	agg = Aggregator()

	start = time.time()
	processed = 0

	for shaft in range(1, num_shafts + 1):
		batch_frames = []
		batch_meta = []

		for cam_idx, cam_id in enumerate(core_config.CAMERA_LIST):
			cam_imgs = cams_images[cam_idx]
			if not cam_imgs:
				continue

			for f_idx in range(frames_per_shaft):
				img_path = cam_imgs[(shaft + f_idx) % len(cam_imgs)]
				img = cv2.imread(img_path)
				if img is None:
					continue
				batch_frames.append(img)
				meta = {core_config.META_CAM_FIELD: cam_id, core_config.META_SHAFT_FIELD: shaft}
				batch_meta.append(meta)

		if not batch_frames:
			continue

		scores, maps = infer.infer_batch(batch_frames, img_size=core_config.IMG_SIZE)

		for sc, me in zip(scores, batch_meta):
			shaft_id = me.get(core_config.META_SHAFT_FIELD, None)
			agg.add(shaft_id, sc, me)
			processed += 1

		decision, info = agg.decide(shaft)
		print(f"Shaft {shaft}: {decision} | info={info}")

	elapsed = time.time() - start
	print(f"Processed frames: {processed}, elapsed={elapsed:.2f}s, avg fps={processed/elapsed if elapsed>0 else 0:.2f}")


if __name__ == '__main__':
	simulate_run(num_shafts=10, frames_per_shaft=1)


