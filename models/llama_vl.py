# models/llama.py

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
#
from models.base_model import BaseModel

class LlamaVLModel(BaseModel):
    def __init__(self, model_name="Llama-3.2-11B-Vision"):
        """
        """
        super().__init__(model_name=model_name)  
        model_id = f"meta-llama/{model_name}-Instruct"
        self.model_id = model_id
        self.device = "cuda"
        
        # 
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def call(self, prompt, **kwargs):
        """
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if len(prompt) > 1:
            image_path = prompt[0]["image"]
            image = Image.open(image_path)
            #
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
        else: # image = ""
            inputs = self.processor(
                None,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
        


        output = self.model.generate(**inputs, max_new_tokens=kwargs.get("max_tokens", 1024))
        # 
        response = self.processor.decode(output[0])
        return prompt, response