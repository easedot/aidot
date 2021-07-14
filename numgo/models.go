package numgo

import (
	"gonum.org/v1/gonum/mat"
)

type ActiveFunc func(x *Variable)*Variable
type Apply func(_,_ int,v float64) float64

type IUpdateGrad interface {
	updateGrad(v *Variable)
}
type IModel interface {
	forward(x *Variable) *Variable
	getLayer()[]*Layer
}

type Model struct {
	IModel IModel
	IUpdateGrad IUpdateGrad
	layers []*Layer
}
func (m *Model) Plot(x *Variable,verbos bool,file string) {
	y:=m.IModel.forward(x)
	y.Plot(verbos,file)
}
func (m *Model) Forward(x *Variable) *Variable {
	return m.IModel.forward(x)
}
func (m *Model) Grad2Param(){
	for _,l:=range m.IModel.getLayer(){
		for _,v:=range l.GetParams(){
			m.IUpdateGrad.updateGrad(v)
		}
	}
}
func (m *Model) ClearGrad(){
	for _,l:=range m.IModel.getLayer(){
		for _,v:=range l.GetParams(){
			v.ClearGrade()
		}
	}
}

func MLP(active ActiveFunc,opt IUpdateGrad,outSizes ...int) *Model{
	mlp:= mLP{}
	if active==nil{
		mlp.Active=Sigmoid
	}
	//for _,o:=range outSizes{
	//	l:=NewLinear(0,o,true)
	//	mlp.layers=append(mlp.Model.layers,l)
	//}
	l:=NewLinear(0,10,true,Sigmoid)
	mlp.layers=append(mlp.Model.layers,l)
	l1:=NewLinear(0,1,true,nil)
	mlp.layers=append(mlp.Model.layers,l1)

	return &Model{IModel: mlp,IUpdateGrad: opt}
}
type mLP struct {
	Model
	Active ActiveFunc
}
func (t mLP) getLayer()[]*Layer{
	return t.layers
}

func (t mLP) forward(x *Variable) *Variable {
	y:=x
	for _,l:=range t.layers{
		y=l.Forward(y)
	}
	return y
}

func SGD(lr float64) IUpdateGrad{
	sgd:=sGD{Lr:lr}
	sgd.apply=func(_,_ int,v float64) float64{return v*lr}
	return &sgd
}
type sGD struct {
	Lr float64
	apply Apply
}
func (s sGD) updateGrad(v *Variable) {
	wt:=mat.Dense{}
	wt.Apply(s.apply,v.Grad.Data)
	v.Data.Sub(v.Data,&wt)
}