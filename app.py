import spaces
import gradio as gr
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = "Write an imperative AI prompt that thoroughly describes the image provided. The prompt makes a text-to-image model generate an exact replica of the source image without being provided the source image. Prompt consists of 4 sentences."

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto").to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID)


@spaces.GPU
def generate(image, max_new_tokens):
    if image is None:
        raise gr.Error("Upload an image.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


with gr.Blocks(title="Prompt Reverse Engineer") as demo:
    gr.Markdown("# Prompt Reverse Engineer")
    gr.Markdown("## Image to Prompt")
    gr.Markdown("Turn an image into a text-to-image prompt with Qwen3-VL.")

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Image")
            max_tokens = gr.Slider(32, 1024, value=214, step=1, label="Max Tokens")
            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column():
            generated_text = gr.Textbox(
                label="Generated Text", lines=12, interactive=False, buttons=["copy"]
            )
            gr.Markdown("Author: Larrain, G")
            
            with gr.Accordion("References", open=False):
                gr.Markdown("""
            @misc{qwen3technicalreport,
                title={Qwen3 Technical Report},
                author={Qwen Team},
                year={2025},
                eprint={2505.09388},
                archivePrefix={arXiv},
                primaryClass={cs.CL},
                url={https://arxiv.org/abs/2505.09388},
            }

            @article{Qwen2.5-VL,
            title={Qwen2.5-VL Technical Report},
            author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
            journal={arXiv preprint arXiv:2502.13923},
            year={2025}
            }

            @article{Qwen2VL,
            title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
            author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
            journal={arXiv preprint arXiv:2409.12191},
            year={2024}
            }

            @article{Qwen-VL,
            title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
            author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
            journal={arXiv preprint arXiv:2308.12966},
            year={2023}
            }
""")

    generate_button.click(generate, [image, max_tokens], generated_text)

if __name__ == "__main__":
    demo.launch()
