package numgo

import (
	"math"

	nt "test_ai/tensor"
)

type ILayer interface {
	forward(x *Variable) *Variable
	getParams() map[string]*Variable
	setParams(map[string]*Variable)
	resetState()
}

func NewLayer(layer ILayer) *Layer {
	return &Layer{ILayer: layer}
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
func (l *Layer) SetParams(p map[string]*Variable) {
	l.ILayer.setParams(p)
}
func (l *Layer) ResetState() {
	l.ILayer.resetState()
}
func NewLinear(inSize, outSize int, haveBias bool, active ActiveFunc) *Layer {
	layer := &linearLayer{}
	layer.Params = make(map[string]*Variable)
	layer.O = outSize
	if active != nil {
		layer.active = active
	}
	layer.W = &Variable{Name: "W"}
	if inSize != 0 {
		layer.I = inSize
		layer.initW()
	}

	if haveBias {
		zb := AsVar(nt.NewZeros(1, layer.O))
		zb.Name = "b"
		layer.b = zb
		layer.Params["b"] = layer.b
	}
	layer.Params["w"] = layer.W
	l := NewLayer(layer)
	return l
}

type linearLayer struct {
	Layer
	active ActiveFunc
	W, b   *Variable
	I, O   int
}

func (l *linearLayer) resetState() {
	panic("implement me")
}

func (l *linearLayer) initW() {
	var InitWFunc = func(v float64) float64 {
		return v * math.Sqrt(1/float64(l.I))
	}
	//var InitWFunc=func(_,_ int,v float64) float64{return v*0.01}
	w := NewVariable(nt.NewRandNorm(l.I, l.O))
	w.Data.Apply(InitWFunc)
	l.W.Data = w.Data
}
func (l *linearLayer) getParams() map[string]*Variable {
	return l.Params
}
func (l *linearLayer) setParams(p map[string]*Variable) {
	l.Params = p
}
func (l *linearLayer) forward(x *Variable) *Variable {
	if l.W.Data == nil {
		l.I = x.Data.Shape()[1]
		l.initW()
	}
	y := Linear(x, l.W, l.b)
	if l.active != nil {
		y = l.active(y)
	}
	return y
}

func NewRNN(inSize, outSize int) *Layer {
	x2h := NewLinear(inSize, outSize, true, nil)
	h2h := NewLinear(inSize, outSize, false, nil)
	r := &rnn{x2h: x2h, h2h: h2h}
	l := NewLayer(r)
	return l
}

type rnn struct {
	Layer
	x2h, h2h *Layer
	h        *Variable
}

func (r *rnn) forward(x *Variable) *Variable {
	hNew := &Variable{}
	if r.h == nil {
		hNew = Tanh(r.x2h.Forward(x))
	} else {
		hNew = Tanh(Add(r.x2h.Forward(x), r.h2h.Forward(r.h)))
	}
	r.h = hNew
	return hNew
}
func (r *rnn) resetState() {
	r.h = nil
}
func (l *rnn) getParams() map[string]*Variable {
	return l.Params
}
func (l *rnn) setParams(p map[string]*Variable) {
	l.Params = p
}

func NewLSTM(inSize, hiddenSize int) *Layer {
	x2f := NewLinear(inSize, hiddenSize, true, nil)
	x2i := NewLinear(inSize, hiddenSize, true, nil)
	x2o := NewLinear(inSize, hiddenSize, true, nil)
	x2u := NewLinear(inSize, hiddenSize, true, nil)

	h2f := NewLinear(hiddenSize, hiddenSize, false, nil)
	h2i := NewLinear(hiddenSize, hiddenSize, false, nil)
	h2o := NewLinear(hiddenSize, hiddenSize, false, nil)
	h2u := NewLinear(hiddenSize, hiddenSize, false, nil)

	r := &lstm{x2f: x2f, x2i: x2i, x2o: x2o, x2u: x2u, h2f: h2f, h2i: h2i, h2o: h2o, h2u: h2u}
	r.resetState()
	l := NewLayer(r)
	return l
}

type lstm struct {
	Layer
	x2f, x2i, x2o, x2u, h2f, h2i, h2o, h2u *Layer
	h, c                                   *Variable
}

func (r *lstm) forward(x *Variable) *Variable {
	hNew, cNew, f, i, o, u := &Variable{}, &Variable{}, &Variable{}, &Variable{}, &Variable{}, &Variable{}
	if r.h == nil {
		f = Sigmoid(r.x2f.Forward(x))
		i = Sigmoid(r.x2i.Forward(x))
		o = Sigmoid(r.x2o.Forward(x))
		u = Tanh(r.x2u.Forward(x))
	} else {
		//todo 这里把add替换成nd.Add
		f = Sigmoid(Add(r.x2f.Forward(x), r.h2f.Forward(r.h)))
		i = Sigmoid(Add(r.x2i.Forward(x), r.h2i.Forward(r.h)))
		o = Sigmoid(Add(r.x2o.Forward(x), r.h2o.Forward(r.h)))
		u = Tanh(Add(r.x2u.Forward(x), r.h2u.Forward(r.h)))
	}
	if r.c == nil {
		cNew = Mul(i, u)
	} else {
		cNew = Add(Mul(f, r.c), Mul(i, u))
	}
	hNew = Mul(o, Tanh(cNew))
	r.h, r.c = hNew, cNew
	return hNew
}

func (r *lstm) resetState() {
	r.h = nil
	r.c = nil
}

func (l *lstm) getParams() map[string]*Variable {
	return l.Params
}
func (l *lstm) setParams(p map[string]*Variable) {
	l.Params = p
}
