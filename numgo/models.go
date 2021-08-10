package numgo

import (
	"log"

	ut "test_ai/utils"
)

type ActiveFunc func(x *Variable) *Variable
type Apply func(_, _ int, v float64) float64

type IUpdateGrad interface {
	updateGrad(v *Variable)
}
type IModel interface {
	forward(x *Variable) *Variable
	getLayer() []*Layer
}

type Model struct {
	IModel      IModel
	IUpdateGrad IUpdateGrad
	layers      []*Layer
}

func (m *Model) Plot(x *Variable, verbos bool, file string) {
	y := m.IModel.forward(x)
	y.Plot(verbos, file)
}
func (m *Model) Forward(x interface{}) *Variable {
	xv := AsVar(x)
	return m.IModel.forward(xv)
}
func (m *Model) SaveWeights(filename string) {
	var p []map[string]*Variable
	for _, l := range m.IModel.getLayer() {
		pv := l.GetParams()
		p = append(p, pv)
	}
	if err := ut.WriteGob(filename, p); err != nil {
		log.Printf("save weight params error")
		return
	}
}
func (m *Model) LoadWeights(filename string) {
	var p []map[string]*Variable
	if err := ut.WriteGob(filename, p); err != nil {
		log.Printf("save weight params error")
		return
	}
	for i, l := range m.IModel.getLayer() {
		l.SetParams(p[i])
	}
}
func (m *Model) Grad2Param() {
	for _, l := range m.IModel.getLayer() {
		for _, v := range l.GetParams() {
			m.IUpdateGrad.updateGrad(v)
		}
	}
}
func (m *Model) ClearGrad() {
	for _, l := range m.IModel.getLayer() {
		for _, v := range l.GetParams() {
			v.ClearGrade()
		}
	}
}

func (m *Model) ResetState() {
	for _, l := range m.IModel.getLayer() {
		l.ResetState()
	}
}
func MLP(active ActiveFunc, opt IUpdateGrad, outSizes ...int) *Model {
	mlp := mLP{}
	if active == nil {
		mlp.Active = Sigmoid
	} else {
		mlp.Active = active
	}
	for i, o := range outSizes {
		var l *Layer
		if i%2 == 0 {
			l = NewLinear(0, o, true, mlp.Active)
		} else {
			l = NewLinear(0, o, true, nil)
		}
		//l=NewLinear(0,o,true,mlp.Active)
		mlp.layers = append(mlp.Model.layers, l)
	}
	return &Model{IModel: &mlp, IUpdateGrad: opt}
}

type mLP struct {
	Model
	Active ActiveFunc
}

func (t *mLP) getLayer() []*Layer {
	return t.layers
}

func (t *mLP) forward(x *Variable) *Variable {
	y := x
	for _, l := range t.layers {
		y = l.Forward(y)
	}
	return y
}

func NewSampleRNN(hiddenSize, outSize int, opt IUpdateGrad) *Model {
	smrnn := simpleRNN{
		rnn: NewRNN(0, hiddenSize),
		fc:  NewLinear(0, outSize, true, nil),
	}
	return &Model{IModel: &smrnn, IUpdateGrad: opt}
}

type simpleRNN struct {
	Model
	rnn, fc *Layer
}

func (s *simpleRNN) ResetState() {
	s.rnn.ResetState()
}
func (s *simpleRNN) getLayer() []*Layer {
	return s.layers
}
func (s *simpleRNN) forward(x *Variable) *Variable {
	y := s.rnn.Forward(x)
	y = s.fc.Forward(y)
	return y
}

func NewBetterRNN(hiddenSize, outSize int, opt IUpdateGrad) *Model {
	btrnn := betterRNN{
		lstm: NewLSTM(0, hiddenSize),
		fc:   NewLinear(0, outSize, true, nil),
	}
	return &Model{IModel: &btrnn, IUpdateGrad: opt}
}

type betterRNN struct {
	Model
	lstm, fc *Layer
}

func (s *betterRNN) ResetState() {
	s.lstm.ResetState()
}
func (s *betterRNN) getLayer() []*Layer {
	return s.layers
}
func (s *betterRNN) forward(x *Variable) *Variable {
	y := s.lstm.Forward(x)
	y = s.fc.Forward(y)
	return y
}
