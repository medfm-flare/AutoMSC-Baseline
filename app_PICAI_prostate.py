import os
import uuid

os.environ.setdefault("nnUNet_raw", "/tmp")
os.environ.setdefault("nnUNet_preprocessed", "/tmp")
os.environ.setdefault("nnUNet_results", "/tmp")

import gradio as gr
from baseline_infer import infer_dataset007_picai

MODALITIES = ["T2W", "ADC", "HBV"]
EXAMPLES = [
    "examples/Dataset007_PICAI/imagesTr/10000_1000000_0000.nii.gz",
    "examples/Dataset007_PICAI/imagesTr/10000_1000000_0001.nii.gz",
    "examples/Dataset007_PICAI/imagesTr/10000_1000000_0002.nii.gz",
]
OUTPUT_BASE = "./gradio_output/dataset007"
os.makedirs(OUTPUT_BASE, exist_ok=True)


def run_inference(f0, f1, f2):
    files = [f for f in [f0, f1, f2] if f is not None]
    if len(files) < 3:
        missing = MODALITIES[len(files):]
        raise gr.Error(f"Missing files: {', '.join(missing)}")

    output_dir = os.path.join(OUTPUT_BASE, str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)

    seg_path, video_path, cls_results = infer_dataset007_picai(
        image=files, output_dir=output_dir, device="cuda",
    )

    task_name = list(cls_results.keys())[0]
    md = f"### {task_name}\n\n| Class | Probability |\n|-------|-------------|\n"
    for cls_name, prob in sorted(cls_results[task_name].items(), key=lambda x: -x[1]):
        md += f"| {cls_name} | {prob:.4f} |\n"

    return video_path, seg_path, md


def load_example():
    return EXAMPLES


with gr.Blocks(title="PICAI Prostate") as demo:
    gr.Markdown("# PICAI Prostate")
    gr.Markdown("Prostate segmentation + ISUP grade classification (Grade 0–5)  \nUpload 3 MRI channels: **T2W, ADC, HBV**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Upload NIfTI files**")
            f0 = gr.File(label="Channel 0 — T2W  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            f1 = gr.File(label="Channel 1 — ADC  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            f2 = gr.File(label="Channel 2 — HBV  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            with gr.Row():
                example_btn = gr.Button("Load Example", variant="secondary")
                run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Overlay Video", format="mp4", height=420, width="100%")
            seg_out = gr.File(label="Download Segmentation Mask (.nii.gz)")
            cls_out = gr.Markdown(label="Classification Results")

    example_btn.click(fn=load_example, inputs=[], outputs=[f0, f1, f2])
    run_btn.click(fn=run_inference, inputs=[f0, f1, f2], outputs=[video_out, seg_out, cls_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, theme=gr.themes.Soft())
