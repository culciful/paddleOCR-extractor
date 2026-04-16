from pathlib import Path
import os

import paddle
from paddleocr import PaddleOCR

input_file = Path("./output_42.pdf")
output_dir = Path("./output")
output_txt = output_dir / f"{input_file.stem}.txt"

print("当前工作目录:", os.getcwd())
print("文件是否存在:", input_file.exists())

if not input_file.exists():
    raise FileNotFoundError(f"未找到输入文件: {input_file.resolve()}")

device = "gpu" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
print("OCR设备(初始):", device)


def build_ocr(target_device: str) -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=target_device,
    )


try:
    ocr = build_ocr(device)
except RuntimeError as err:
    err_msg = str(err)
    # RTX 50 系列在旧版 Paddle 包上可能触发 "Unsupported GPU architecture"。
    if device == "gpu" and "Unsupported GPU architecture" in err_msg:
        print("检测到当前 Paddle 暂不支持此 GPU 架构，自动切换到 CPU 继续。")
        device = "cpu"
        ocr = build_ocr(device)
    else:
        raise

print("OCR设备(实际):", device)

results = ocr.predict(str(input_file))

output_dir.mkdir(parents=True, exist_ok=True)
lines = []

for page_idx, res in enumerate(results, start=1):
    res.save_to_json("output")
    res.save_to_img("output")
#     lines.append(f"===== Page {page_idx} =====\n")
#     rec_texts = getattr(res, "rec_texts", None)
#     if rec_texts:
#         for text in rec_texts:
#             lines.append(f"{text}\n")
#     lines.append("\n")

# with open(output_txt, "w", encoding="utf-8") as f:
#     f.writelines(lines)

# print(f"识别完成，已保存: {output_txt.resolve()}")