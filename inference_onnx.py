import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort


def normalize(frame: np.ndarray) -> np.ndarray:
    """
    Args:
        frame: BGR
    Returns: normalized 0~1 BCHW RGB
    """
    img = frame.astype(np.float32).copy() / 255.0
    img = img[:, :, ::-1]  # RGB
    img = np.transpose(img, (2, 0, 1))  # (C,H,W)
    img = np.expand_dims(img, axis=0)  # (B=1,C,H,W)
    return img


def infer_rvm_frame(weight: str = "rvm_mobilenetv3_fp32.onnx",
                    img_path: str = "test.jpg",
                    output_path: str = "test_onnx.jpg"):
    sess = ort.InferenceSession(f'./checkpoint/{weight}')
    print(f"Load checkpoint/{weight} done!")

    for _ in sess.get_inputs():
        print("Input: ", _)
    for _ in sess.get_outputs():
        print("Input: ", _)

    frame = cv2.imread(img_path)
    src = normalize(frame)
    rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4  # 必须用模型一样的 dtype
    downsample_ratio = np.array([0.25], dtype=np.float32)  # 必须是 FP32
    bgr = np.array([0.47, 1., 0.6]).reshape((3, 1, 1))

    fgr, pha, *rec = sess.run([], {
        'src': src,
        'r1i': rec[0],
        'r2i': rec[1],
        'r3i': rec[2],
        'r4i': rec[3],
        'downsample_ratio': downsample_ratio
    })

    merge_frame = fgr * pha + bgr * (1. - pha)  # (1,3,H,W)
    merge_frame = merge_frame[0] * 255.  # (3,H,W)
    merge_frame = merge_frame.astype(np.uint8)  # RGB
    merge_frame = np.transpose(merge_frame, (1, 2, 0))  # (H,W,3)
    merge_frame = cv2.cvtColor(merge_frame, cv2.COLOR_BGR2RGB)

    cv2.imwrite(output_path, merge_frame)

    print(f"infer done! saved {output_path}")


def infer_rvm_video(weight: str = "rvm_mobilenetv3_fp32.onnx",
                    video_path: str = "./demo/1917.mp4",
                    output_path: str = "./demo/1917_onnx.mp4"):
    sess = ort.InferenceSession(f'./checkpoint/{weight}')
    print(f"Load checkpoint/{weight} done!")

    for _ in sess.get_inputs():
        print("Input: ", _)
    for _ in sess.get_outputs():
        print("Input: ", _)

    # 读取视频
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Caputer: Height: {height}, Width: {width}, Frame Count: {frame_count}")

    # 写出视频
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Create Video Writer: {output_path}")

    i = 0
    rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4  # 必须用模型一样的 dtype
    downsample_ratio = np.array([0.25], dtype=np.float32)  # 必须是 FP32
    bgr = np.array([0.47, 1., 0.6]).reshape((3, 1, 1))

    print(f"Infer {video_path} start ...")
    while video_capture.isOpened():
        success, frame = video_capture.read()

        if success:
            i += 1
            src = normalize(frame)
            # src 张量是 [B, C, H, W] 形状，必须用模型一样的 dtype
            t1 = time.time()
            fgr, pha, *rec = sess.run([], {
                'src': src,
                'r1i': rec[0],
                'r2i': rec[1],
                'r3i': rec[2],
                'r4i': rec[3],
                'downsample_ratio': downsample_ratio
            })
            t2 = time.time()
            print(f"Infer {i}/{frame_count} done! -> cost {(t2 - t1) * 1000} ms", end=" ")
            merge_frame = fgr * pha + bgr * (1. - pha)  # (1,3,H,W)
            merge_frame = merge_frame[0] * 255.  # (3,H,W)
            merge_frame = merge_frame.astype(np.uint8)  # RGB
            merge_frame = np.transpose(merge_frame, (1, 2, 0))  # (H,W,3)
            merge_frame = cv2.cvtColor(merge_frame, cv2.COLOR_BGR2RGB)
            merge_frame = cv2.resize(merge_frame, (width, height))

            video_writer.write(merge_frame)
            print(f"write {i}/{frame_count} done.")
        else:
            print("can not read video! skip!")
            break

    video_capture.release()
    video_writer.release()
    print(f"Infer {video_path} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="video")
    parser.add_argument("--weight", type=str, default="rvm_mobilenetv3_fp32.onnx")
    parser.add_argument("--input", type=str, default="./demo/1917.mp4")
    parser.add_argument("--output", type=str, default="./demo/1917_onnx.mp4")
    args = parser.parse_args()

    if args.mode == "video":
        infer_rvm_video(weight=args.weight, video_path=args.input, output_path=args.output)
    else:
        infer_rvm_frame(weight=args.weight, img_path=args.input, output_path=args.output)

    """
    PYTHONPATH=. python3 ./inference_onnx.py --input ./demo/1917.mp4 --output ./demo/1917_onnx.mp4
    PYTHONPATH=. python3 ./inference_onnx.py --mode img --input test.jpg --output test_onnx.jpg
    """
