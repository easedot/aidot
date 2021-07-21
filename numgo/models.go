package numgo

import (
	"math"

	"gonum.org/v1/gonum/mat"
	ut "test_ai/utils"
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
	}else{
		mlp.Active=active
	}
	for i,o:=range outSizes{
		var l *Layer
		if i%2==0{
			//l=NewLinear(0,o,true,nil)
		}else{
			//l=NewLinear(0,o,true,mlp.Active)
		}
		l=NewLinear(0,o,true,mlp.Active)
		mlp.layers=append(mlp.Model.layers,l)
	}
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
	sgd.apply=func(_,_ int,v float64) float64{
		return v*lr
	}
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


func Adam(alpha,beta1,beta2,eps float64) IUpdateGrad {
	adm:=&aDam{
		t:0,
		alpha: alpha,
		beta1: beta1,
		beta2: beta2,
		ms:make(map[*Variable]*Variable),
		vs:make(map[*Variable]*Variable),
		eps: eps,
	}
	adm.applyLr=func(_,_ int,v float64) float64{return v*adm.Lr()}
	adm.applySqrt=func(_,_ int,v float64) float64{return math.Sqrt(v)+eps}
	return adm
}

type aDam struct {
	t float64
	alpha,beta1,beta2,eps float64
	ms map[*Variable]*Variable
	vs map[*Variable]*Variable
	applyLr Apply
	applySqrt Apply
}

func (a *aDam) Lr() float64 {
	fix1:=1-math.Pow(a.beta1,a.t)
	fix2:=1-math.Pow(a.beta2,a.t)
	return a.alpha*math.Sqrt(fix2)/fix1
}
func (a *aDam) updateGrad(param *Variable) {
	a.t+=1
	if _,ok:=a.ms[param];!ok{
		a.ms[param]=&Variable{Data:ut.LikeZeros(param.Data)}
		a.vs[param]=&Variable{Data:ut.LikeZeros(param.Data)}
	}
	m, v :=a.ms[param],a.vs[param]
	g:=param.Grad
	m=Add(m,Mul(NewVar(1-a.beta1),Sub(g,m)))
	v=Add(v,Mul(NewVar(1+a.beta2),Sub(Mul(g,g),v)))
	m.Data.Apply(a.applyLr,m.Data)
	v.Data.Apply(a.applySqrt,v.Data)

	param.Data=Sub(param,Div(m,v)).Data
}
