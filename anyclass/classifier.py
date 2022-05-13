"""
	im so lazy i just needed something to quickly make a classifier even though i only use stuff like it
	once or twice a month 
	this is DEFINITELY overkill but i dont CARE
"""
import clip
from torch import nn
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm.auto import trange
import io 
import requests  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Prediction:
	def __init__(self, classes, outputs):
		self.classes = classes
		self.outputs = outputs
	def to_string(self):
		assert self.outputs.shape[0]==1, "to_string only supported for single image predictions"
		predictions = list(zip(self.classes, self.outputs.tolist()[0]))
		to_string = lambda x,y: f"{x}    {y}%"
		predictions = [to_string(prediction[0], prediction[1]) for prediction in predictions]
		predictions = "\n".join(predictions)
		return predictions

class SphericalDistanceLoss(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x, y):
		x = F.normalize(x.float(), dim=-1)
		y = F.normalize(y.float(), dim=-1)
		l = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
		return l  
sph_dist = SphericalDistanceLoss()

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        projection_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Classifier(nn.Module):
	def __init__(self, categories, clip_model="ViT-B/32", projection=False, projection_dim=512):
		super().__init__()
		self.categories = categories
		self.model = clip_model
		self.model = clip.load(clip_model, jit=False)[0].to(device).float()
		self.fix = transforms.Resize((self.model.visual.input_resolution,self.model.visual.input_resolution))
		if projection:
			self.embedding_dim = 512 if "l/" not in clip_model else 768
			self.projection_dim=512
			self.projection_head = ProjectionHead(embedding_dim=self.embedding_dim, projection_dim=self.projection_dim)
		else:
			self.projection_head = nn.Identity()
		self.projection_head = self.projection_head.to(device)
		self.embeddings = [self.projection_head(self.model.encode_text(clip.tokenize(x).to(device))) for x in categories]
	def forward(self, x):
		if "tensor" not in str(type(x)).lower() and "list" not in str(type(x)).lower():
			x = transforms.ToTensor()(x).unsqueeze(0)
			x = self.fix(x).to(device)
		if str(type(x))=="<class 'list'>":
			x = [transforms.ToTensor()(i) for i in x]
			x = torch.cat([self.fix(i).unsqueeze(0) for i in x], 0).to(device)
		x = self.model.encode_image(x)
		logits = self.projection_head(x)
		logits = 1-torch.cat([torch.cat([sph_dist(i, logits[j]).unsqueeze(0) for i in self.embeddings]).unsqueeze(0) for j in range(logits.shape[0])], 0)
		
		softmax = F.softmax(logits, dim=-1)
		return Prediction(self.categories, softmax)
	def recompute_classes(self):
		self.embeddings = [self.projection_head(self.model.encode_text(clip.tokenize(x).to(device))) for x in self.categories]


def load_img(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return Image.open(fd)
    return Image.open(open(url_or_path, 'rb'))