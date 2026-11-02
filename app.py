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

    generate_button.click(generate, [image, max_tokens], generated_text)

if __name__ == "__main__":
    demo.launch()
