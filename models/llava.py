import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from models.base_model import BaseModel


class LlavaModel(BaseModel):
	def __init__(self, model_name="llava-v1.6-mistral-7b"):
		super().__init__(model_name=model_name)  
		self.model_id = f"llava-hf/{model_name}-hf"
		self.device = "cuda" # "cuda:0"
  
		self.processor = LlavaNextProcessor.from_pretrained(self.model_id)

		self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
		self.model.to("cuda:0")
		
  
	def call(self, prompt, **kwargs):

		messages = [
			{"role": "user", "content": prompt},
		]
  
		input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
		
		if len(prompt) > 1:
			image_path = prompt[0]["image"]
			image = Image.open(image_path)
			#
			inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(self.model.device)
			
		else: # image = ""
			inputs = self.processor(images=None, text=input_text, return_tensors="pt").to(self.model.device)
			
		#
		output = self.model.generate(**inputs, max_new_tokens=kwargs.get("max_tokens", 1024))
		# 
		response = self.processor.decode(output[0], skip_special_tokens=True)
		return prompt, response