import os
import uuid

os.environ.setdefault("nnUNet_raw", "/tmp")
os.environ.setdefault("nnUNet_preprocessed", "/tmp")
os.environ.setdefault("nnUNet_results", "/tmp")

import gradio as gr
from baseline_infer import infer_dataset005_mu_glioma_post

MODALITIES = ["t1c", "t1n", "t2f", "t2w"]
EXAMPLES = [
    "examples/Dataset005_MU_Glioma_Post/imagesTr/Pat0003_TP1_0000.nii.gz",
    "examples/Dataset005_MU_Glioma_Post/imagesTr/Pat0003_TP1_0001.nii.gz",
    "examples/Dataset005_MU_Glioma_Post/imagesTr/Pat0003_TP1_0002.nii.gz",
    "examples/Dataset005_MU_Glioma_Post/imagesTr/Pat0003_TP1_0003.nii.gz",
]
OUTPUT_BASE = "./gradio_output/dataset005"
os.makedirs(OUTPUT_BASE, exist_ok=True)


def run_inference(f0, f1, f2, f3):
    files = [f for f in [f0, f1, f2, f3] if f is not None]
    if len(files) < 4:
        missing = MODALITIES[len(files):]
        raise gr.Error(f"Missing files: {', '.join(missing)}")

    output_dir = os.path.join(OUTPUT_BASE, str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)

    seg_path, video_path, cls_results = infer_dataset005_mu_glioma_post(
        image=files, output_dir=output_dir, device="cuda",
    )

    task_name = list(cls_results.keys())[0]
    md = f"### {task_name}\n\n| Class | Probability |\n|-------|-------------|\n"
    for cls_name, prob in sorted(cls_results[task_name].items(), key=lambda x: -x[1]):
        md += f"| {cls_name} | {prob:.4f} |\n"

    return video_path, seg_path, md


def load_example():
    return EXAMPLES


with gr.Blocks(title="MU Glioma Post") as demo:
    gr.Markdown("# MU Glioma Post")
    gr.Markdown("Brain Glioma segmentation + classification (GBM / Astrocytoma / Others)  \nUpload 4 MRI channels: **t1c, t1n, t2f, t2w**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Upload NIfTI files**")
            f0 = gr.File(label="Channel 0 — t1c  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            f1 = gr.File(label="Channel 1 — t1n  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            f2 = gr.File(label="Channel 2 — t2f  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            f3 = gr.File(label="Channel 3 — t2w  (.nii / .nii.gz)", file_types=[".gz", ".nii"])
            with gr.Row():
                example_btn = gr.Button("Load Example", variant="secondary")
                run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Overlay Video", format="mp4", height=420, width="100%")
            seg_out = gr.File(label="Download Segmentation Mask (.nii.gz)")
            cls_out = gr.Markdown(label="Classification Results")

    example_btn.click(fn=load_example, inputs=[], outputs=[f0, f1, f2, f3])
    run_btn.click(fn=run_inference, inputs=[f0, f1, f2, f3], outputs=[video_out, seg_out, cls_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, theme=gr.themes.Soft())
