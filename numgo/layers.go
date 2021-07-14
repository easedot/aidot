package numgo

import (
	"math"
)

type ILayer interface {
	forward(x *Variable) *Variable
	getParams()map[string]*Variable
}

func NewLayer(layer ILayer)*Layer{
	return &Layer{ILayer: layer }
}

type Layer struct {
	ILayer ILayer
	Params map[string]*Variable
}

func (l *Layer) Forward(x *Variable) *Variable {
	return l.ILayer.forward(x)
}
func (l *Layer) GetParams() map[string]*Variable {
	return l.ILayer.getParams()
}

func NewLinear(inSize,outSize int,haveBias bool,active ActiveFunc) *Layer {
	layer := &linearLayer{}
	layer.Params=make(map[string]*Variable)
	layer.O=outSize
	if active!=nil{
		layer.active=active
	}
	if inSize!=0{
		layer.I=inSize
		layer.initW()
	}
	layer.W =&Variable{Name: "W"}

	if haveBias{
		zb := NewZeros(1, layer.O)
		zb.Name="b"
		layer.b = zb
		layer.Params["b"]=layer.b
	}
	layer.Params["w"]=layer.W
	l:= NewLayer(layer)
	return l
}

type linearLayer struct {
	Layer
	active ActiveFunc
	W,b *Variable
	I,O int
}
func (l *linearLayer) initW(){
	var InitWFunc=func(_,_ int,v float64) float64{return v*math.Sqrt(1/float64(l.I))}
	//var InitWFunc=func(_,_ int,v float64) float64{return v*0.01}
	w := NewRandN(l.I, l.O)
	w.Data.Apply(InitWFunc,w.Data)
	l.W.Data=w.Data
}
func (l *linearLayer) getParams()map[string]*Variable{
	return l.Params
}
func (l *linearLayer) forward(x *Variable) *Variable {
	if l.W.Data==nil{
		l.I=x.Shape().C
		l.initW()
	}
	y:=Linear(x,l.W,l.b)
	if l.active!=nil{
		y=l.active(y)
	}
	return y
}
