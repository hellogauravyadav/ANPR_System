from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from anpr.config import load_config, resolve
from anpr.logger import PlateLogger
from anpr.pipeline import ANPRPipeline
from anpr.sources import IMAGE_EXT, ThreadedCapture, VideoFileCapture, classify_source
from anpr.visualize import draw_results


def parse_args():
    p = argparse.ArgumentParser(description="Real-time ANPR (YOLO + EasyOCR)")
    p.add_argument("--source", default="0", help="webcam index, image, folder, video file or stream URL")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--weights", help="override detector weights path")
    p.add_argument("--conf", type=float, help="override detector confidence threshold")
    p.add_argument("--imgsz", type=int, help="override inference size")
    p.add_argument("--device", help="auto | cpu | 0")
    p.add_argument("--no-ocr", action="store_true", help="detection only (fastest)")
    p.add_argument("--save", action="store_true", help="save annotated image/video to outputs/")
    p.add_argument("--no-show", action="store_true", help="do not open a preview window")
    p.add_argument("--no-log", action="store_true", help="do not write plate_log.csv / crops")
    return p.parse_args()


def build_overrides(a) -> dict:
    det = {k: v for k, v in {"weights": a.weights, "conf": a.conf, "imgsz": a.imgsz,
                             "device": a.device}.items() if v is not None}
    ov: dict = {"detector": det}
    if a.no_ocr:
        ov["ocr"] = {"enabled": False}
    if a.no_log:
        ov["output"] = {"log_csv": False}
    return ov


def make_writer(path: Path, fps: float, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def print_results(name, results):
    for r in results:
        txt = r.text or "(unread)"
        print(f"  {name}: {txt:<14} score={r.score:.2f} det={r.det_conf:.2f} "
              f"ocr={r.ocr_conf:.2f} valid={r.valid}")


# ----------------------------------------------------------------------
def run_images(pipe, cfg, paths, args, out_dir):
    logger = PlateLogger(out_dir, "images", cfg.output.min_log_score, cfg.output.save_crops,
                         cfg.output.log_csv)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skipping unreadable image: {path}")
            continue
        results = pipe.process_image(frame)
        for tr in pipe.pop_finished():
            logger.log(tr)
        print(f"{Path(path).name}: {len(results)} plate(s)")
        print_results(Path(path).name, results)
        annotated = draw_results(frame, results, cfg.display)
        if args.save:
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{Path(path).stem}_anpr.jpg"), annotated)
        if not args.no_show:
            cv2.imshow("ANPR", annotated)
            if cv2.waitKey(0 if len(paths) == 1 else 800) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    if logger.count:
        print(f"Plates logged: {logger.count} -> {logger.csv_path}")


def run_stream(pipe, cfg, source, kind, args, out_dir):
    cap = (VideoFileCapture(source) if kind == "video"
           else ThreadedCapture(source, cfg.camera.width, cfg.camera.height))
    logger = PlateLogger(out_dir, source, cfg.output.min_log_score, cfg.output.save_crops,
                         cfg.output.log_csv)
    writer = None
    out_name = ("webcam" if kind == "webcam" else Path(source).stem if kind == "video" else "stream")
    out_path = out_dir / f"{out_name}_anpr.mp4"

    fps_smooth, prev, frame_idx = 0.0, time.time(), 0
    print("Running... press q or ESC in the window to quit." if not args.no_show else "Running...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = pipe.process_frame(frame, frame_idx)
            for tr in pipe.pop_finished():
                logger.log(tr)

            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps_smooth = inst if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst
            prev = now

            annotated = draw_results(frame, results, cfg.display, fps_smooth)
            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_writer(out_path, min(cap.fps, 30.0), (w, h))
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("ANPR", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_idx += 1
            if kind == "video" and frame_idx % 100 == 0 and cap.total:
                print(f"  {frame_idx}/{cap.total} frames  ({fps_smooth:.1f} FPS)")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        for tr in pipe.flush():
            logger.log(tr)

    print(f"Processed {frame_idx} frames. Plates logged: {logger.count}")
    if args.save and writer is not None:
        print(f"Annotated video: {out_path}")
    if logger.enabled and logger.count:
        print(f"CSV log: {logger.csv_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config, build_overrides(args))
    out_dir = resolve(cfg.output.dir)
    kind = classify_source(args.source)

    print("Loading models (first run downloads EasyOCR weights)...")
    pipe = ANPRPipeline(cfg)
    print(f"Detector device: {pipe.detector.device} | OCR: {'on' if pipe.ocr else 'off'} | source: {kind}")

    if kind in ("image", "folder"):
        if kind == "image":
            paths = [Path(args.source)]
        else:
            paths = sorted(p for p in Path(args.source).iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not paths:
            raise SystemExit(f"No images found in {args.source}")
        run_images(pipe, cfg, paths, args, out_dir)
    else:
        run_stream(pipe, cfg, args.source, kind, args, out_dir)


if __name__ == "__main__":
    main()
