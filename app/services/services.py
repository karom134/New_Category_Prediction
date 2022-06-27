from fastapi import APIRouter
from pydantic import BaseModel
from ..model_config.model_builder import predict

router = APIRouter(
	prefix="/cat_prediction",
	tags=["Category_Prediction"],
	responses={404:{"description":"Not found"}}
	)

class Product(BaseModel):
	title: str
	desc: str

@router.put("/")
async def MakePrediction(product: Product):
	classes=predict(product.title,product.desc)
	resp={}
	resp_key=0
	for cat_name in classes:
		resp[resp_key]={"category_name":cat_name}
		resp_key+=1
	return resp
