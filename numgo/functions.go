package numgo

import (
	nd "test_ai/numed"
	ut "test_ai/utils"
)

func Sin(x *Variable) *Variable {
	f := NewFunction(&sinFunc{})
	y := f.Run(x)
	return y
}

type sinFunc struct {
	Function
}

func (s *sinFunc) forward(i []*Variable) []*Variable {
	x := i[0]
	y := x.Data.Sin()
	return []*Variable{{Data: y}}
}
func (s *sinFunc) backward(i, _, gy []*Variable) []*Variable {
	x := i[0]
	g := gy[0]
	y := Mul(Cos(x), g)
	return []*Variable{y}
}

func Cos(x *Variable) *Variable {
	f := NewFunction(&cosFunc{})
	y := f.Run(x)
	return y
}

type cosFunc struct {
	Function
}

func (s *cosFunc) forward(i []*Variable) []*Variable {
	x := i[0]
	y := x.Data.Cos()
	return []*Variable{{Data: y}}
}
func (s *cosFunc) backward(i, _, gy []*Variable) []*Variable {
	x := i[0]
	g := gy[0]
	y := Mul(g, Neg(Sin(x)))
	return []*Variable{y}
}

func Tanh(x *Variable) *Variable {
	f := NewFunction(&tanhFunc{})
	y := f.Run(AsVar(x))
	return y
}

type tanhFunc struct {
	Function
}

func (s *tanhFunc) forward(i []*Variable) []*Variable {
	x := i[0]
	y := x.Data.Tanh()
	return []*Variable{{Data: y}}
}
func (s *tanhFunc) backward(_, o, gy []*Variable) []*Variable {
	y := o[0]
	g := gy[0]
	gx := Mul(g, Neg(Sub(NewVar(1), Mul(y, y))))
	return []*Variable{gx}
}

func Reshape(x *Variable, s *nd.Shape) *Variable {
	f := NewFunction(&reshapeFunc{S: s})
	y := f.Run(x)
	return y
}

type reshapeFunc struct {
	Function
	S  *nd.Shape
	Xs *nd.Shape
}

func (s *reshapeFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	s.Xs = x.Shape()
	return []*Variable{{Data: x.Data.Reshape(s.S.R, s.S.C)}}
}
func (s *reshapeFunc) backward(_, _, gy []*Variable) []*Variable {
	return []*Variable{Reshape(gy[0], s.Xs)}
}

func Transpose(x *Variable) *Variable {
	f := NewFunction(&transposeFunc{})
	y := f.Run(x)
	return y
}

type transposeFunc struct {
	Function
}

func (t *transposeFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	return []*Variable{_tranposeTo(x)}
}
func (t *transposeFunc) backward(_, _, gy []*Variable) []*Variable {
	y := gy[0]
	return []*Variable{Transpose(y)}
}

func Exp(x ...*Variable) *Variable {
	f := NewFunction(&expFunc{})
	return f.Run(x...)
}

type expFunc struct {
	Function
}

func (e *expFunc) forward(i []*Variable) []*Variable {
	//todo 这里不能使用dense的计算，因为gonum要求方形矩阵？
	x := i[0]
	return []*Variable{{Data: x.Data.Exp()}}
}
func (e *expFunc) backward(_, o, gy []*Variable) []*Variable {
	y := o[0]
	g := gy[0]
	gx := Mul(g, y)
	return []*Variable{gx}
}

func Sum(x *Variable) *Variable {
	f := NewFunction(&sumFunc{})
	y := f.Run(x)
	return y
}

type sumFunc struct {
	Function
	Xs *nd.Shape
}

func (s *sumFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	s.Xs = x.Shape()
	y := x.Data.Sum(nil, true)
	r := NewVar(y.Var())
	return []*Variable{r}
}
func (s *sumFunc) backward(i, o, gy []*Variable) []*Variable {
	gx := BroadCastTo(gy[0], s.Xs)
	return []*Variable{gx}
}

func SumTo(x *Variable, s *nd.Shape) *Variable {
	f := NewFunction(&sumToFunc{S: s})
	y := f.Run(x)
	return y
}

type sumToFunc struct {
	Function
	S  *nd.Shape
	Xs *nd.Shape
}

func (s *sumToFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	s.Xs = x.Shape()
	y := _sumTo(x, s.S)
	return []*Variable{y}
}
func (s *sumToFunc) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	gx := BroadCastTo(g, s.Xs)
	return []*Variable{gx}
}

func BroadCastTo(x *Variable, s *nd.Shape) *Variable {
	f := NewFunction(&broadcastToFunc{S: s})
	y := f.Run(x)
	return y
}

type broadcastToFunc struct {
	Function
	S  *nd.Shape
	Xs *nd.Shape
}

func (s *broadcastToFunc) forward(ix []*Variable) []*Variable {
	s.Xs = ix[0].Shape()
	y := ix[0]
	if s.S.B(s.Xs) {
		y = _broadcastTo(ix[0], s.S)
	}
	return []*Variable{y}
}
func (s *broadcastToFunc) backward(i, o, gy []*Variable) []*Variable {
	gx := gy[0]
	if s.S.B(s.Xs) {
		gx = SumTo(gy[0], s.Xs)
	}
	return []*Variable{gx}
}

func Matmul(x, w *Variable) *Variable {
	f := NewFunction(&matMulFunc{})
	y := f.Run(x, w)
	return y
}

type matMulFunc struct {
	Function
}

func (s *matMulFunc) forward(ix []*Variable) []*Variable {
	x, W := ix[0], ix[1]
	y := nd.Dot(x.Data, W.Data)
	return []*Variable{{Data: y}}
}
func (s *matMulFunc) backward(i, o, gy []*Variable) []*Variable {
	x, W := i[0], i[1]
	g := gy[0]
	gx := Matmul(g, W.T())
	gW := Matmul(x.T(), g)
	return []*Variable{gx, gW}
}

type linearFunc struct {
	Function
}

func (l *linearFunc) forward(ix []*Variable) []*Variable {
	x, W := ix[0], ix[1]
	y := nd.Dot(x.Data, W.Data)

	var b *Variable
	if len(ix) > 2 {
		b = ix[2]
	}
	if b != nil {
		y = nd.Add(y, b.Data)
	}
	v := &Variable{Data: y}
	return []*Variable{v}
}

func (l *linearFunc) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x, W := is[0], is[1]
	var b *Variable
	if len(is) > 2 {
		b = is[2]
	}
	gb := &Variable{}
	if b != nil {
		gb = SumTo(gy, b.Shape())
	}
	gx := Matmul(gy, W.T())
	gW := Matmul(x.T(), gy)
	return []*Variable{gx, gW, gb}
}
func Linear(x, W, b *Variable) *Variable {
	f := NewFunction(&linearFunc{})
	if b == nil {
		return f.Run(x, W)
	} else {
		return f.Run(x, W, b)
	}
}

type sigmoid struct {
	Function
}

func (s *sigmoid) forward(ix []*Variable) []*Variable {
	x := ix[0]
	//other
	//y:=nd.Add(nd.Mul(nd.Mul(x.Data,0.5).Tanh(),0.5),0.5)
	y := nd.Add(nd.Mul(nd.Tanh(nd.Mul(x.Data, 0.5)), 0.5), 0.5)
	v := &Variable{Data: y}
	return []*Variable{v}
}
func (s *sigmoid) backward(is, os, gys []*Variable) []*Variable {
	y := os[0]
	gy := gys[0]
	gx := Mul(Mul(gy, y), Sub(1, y))
	return []*Variable{gx}
}
func Sigmoid(x *Variable) *Variable {
	f := NewFunction(&sigmoid{})
	return f.Run(x)
}

type meanSquaredError struct {
	Function
}

func (m *meanSquaredError) forward(ix []*Variable) []*Variable {
	x0, x1 := ix[0], ix[1]
	diff := nd.Sub(x0.Data, x1.Data)
	y := nd.Div(diff.Pow(2).Sum(nil, true), diff.Shape().R)
	v := &Variable{Data: y}
	return []*Variable{v}
}
func (m *meanSquaredError) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x0, x1 := is[0], is[1]
	diff := Sub(x0, x1)
	gx0 := Mul(Mul(gy, diff), Div(2, diff.Shape().R))
	gx1 := Neg(gx0)
	return []*Variable{gx0, gx1}
}

func MeanSquaredError(x0, x1 *Variable) *Variable {
	f := NewFunction(&meanSquaredError{})
	return f.Run(x0, x1)
}
func Softmax1d(x *Variable) *Variable {
	y := x.Data.Exp()
	sumY := y.Sum(1, true)
	return Div(y, sumY)
}

func Softmax_cross_entropy_simple(x *Variable, t []int) float64 {
	N := x.Shape().R
	p := Softmax1d(x)
	p = Clip(p, 1e-15, 1.0)
	p = Log(p)
	p = Getitem(p, ut.ArangeInt(0, N, 1), t)
	sum := Sum(p)
	y := -1 * sum.At(0, 0) / float64(N)
	return y
}

func Accuracy(y interface{}, t []int) *Variable {
	yv := AsVar(y)
	tv := NewVecInt(t...)
	pred := yv.Data.ArgMax(1, true).Reshape(tv.Data.Dims())
	mask := pred.Mask(func(i, j int, v float64) float64 {
		if v == tv.Data.Get(i, j) {
			return 1.0
		} else {
			return 0.0
		}
	})
	return NewVar(mask.Mean().Var())
}

func Dropout(xi interface{}, dropRatio float64) *Variable {
	x := AsVar(xi)
	if Backprop { //for train
		mask := NewRand(x.Data.Dims()).Data.Mask(func(i, j int, v float64) float64 {
			if v > dropRatio {
				return 1
			} else {
				return 0
			}
		})
		scale := 1 - dropRatio
		y := Div(Mul(x, mask), scale)
		return y
	} else { //for test
		return x
	}
}

func SoftmaxCrossEntroy(x interface{}, t []int) *Variable {
	xv := AsVar(x)
	f := NewFunction(&softmaxCrossEntropy{t: t})
	y := f.Run(xv)
	return y
}

type softmaxCrossEntropy struct {
	Function
	t []int
}

func (s *softmaxCrossEntropy) forward(is []*Variable) []*Variable {
	x := is[0]
	N, _ := x.Data.Dims()
	logZ := _logsumexp(x, 1)
	logP := nd.Sub(x.Data, logZ.Data) //Sub(x,logZ)
	logP = logP.RowsCol(s.t...)       // SelRowCol(logP.Data,s.t...) //Getitem(logP,ut.ArangeInt(0,N,1),s.t)
	sum := logP.Sum(nil, true).Neg()
	y := sum.Var() / float64(N)
	return []*Variable{NewVar(y)}
}
func (s *softmaxCrossEntropy) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x := is[0]
	N, ClsNum := x.Data.Dims()
	gy = Mul(gy, 1.0/float64(N))
	y := Softmax(x, 1)
	//to one hot
	eyes := _eyes(ClsNum)
	tOneHot := OneHot(eyes, s.t)
	y = Mul(Sub(y, tOneHot), gy)
	return []*Variable{y}
}

func Softmax(x *Variable, axis interface{}) *Variable {
	f := NewFunction(&softmax{axis: axis})
	y := f.Run(x)
	return y
}

type softmax struct {
	Function
	axis interface{}
}

func (s *softmax) forward(is []*Variable) []*Variable {
	x := is[0]
	y := Sub(x, Max(x, s.axis))
	y = Exp(y)
	ym := _sum(y, s.axis, true)
	y = Div(y, ym)
	return []*Variable{y}
}
func (s *softmax) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	y := os[0]
	gx := Mul(y, gy)
	sumdx := _sum(gx, s.axis, true)
	gx = Sub(gx, Mul(y, sumdx))
	return []*Variable{gx}
}

type logSoftmax struct {
	Function
}

func (s *logSoftmax) forward(is []*Variable) []*Variable {
	return []*Variable{}
}
func (s *logSoftmax) backward(is, os, gys []*Variable) []*Variable {
	return []*Variable{}
}

func ReLU(x *Variable) *Variable {
	f := NewFunction(&reLU{})
	y := f.Run(x)
	return y
}

type reLU struct {
	Function
}

func (s *reLU) forward(is []*Variable) []*Variable {
	x := is[0]
	y := nd.Maximum(x.Data, 0.0)
	return []*Variable{{Data: y}}
}
func (s *reLU) backward(is, os, gys []*Variable) []*Variable {
	x := is[0]
	mask := x.Data.Mask(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		} else {
			return 0
		}
	})
	gy := gys[0]
	gx := Mul(gy, mask)
	return []*Variable{gx}
}

type leakyReLU struct {
	Function
}

func (s *leakyReLU) forward(is []*Variable) []*Variable {
	return []*Variable{}
}
func (s *leakyReLU) backward(is, os, gys []*Variable) []*Variable {
	return []*Variable{}
}

func Getitem(x *Variable, rs []int, cs []int) *Variable {
	f := NewFunction(&getItem{rs: rs, cs: cs})
	y := f.Run(x)
	return y
}

type getItem struct {
	Function
	rs []int
	cs []int
}

func (s *getItem) forward(is []*Variable) []*Variable {
	x := is[0]
	var y []float64
	for i := 0; i < len(s.rs); i++ {
		r := s.rs[i]
		c := s.cs[i]
		y = append(y, x.At(r, c))
	}
	return []*Variable{NewVec(y...)}
}

func (s *getItem) backward(is, os, gys []*Variable) []*Variable {
	//x:=is[0]

	return []*Variable{}
}

type GetItemGrad struct {
	Function
}

func Log(x *Variable) *Variable {
	x.Data = x.Data.Log()
	return x
}

func Clip(x *Variable, min, max float64) *Variable {
	f := NewFunction(&clipFunc{min: min, max: max})
	y := f.Run(x)
	return y
}

type clipFunc struct {
	Function
	max, min float64
}

func (cp *clipFunc) forward(is []*Variable) []*Variable {
	x := is[0]
	y := x.Data.Clip(cp.min, cp.max)
	return []*Variable{{Data: y}}
}
func (cp *clipFunc) backward(is, os, gys []*Variable) []*Variable {
	eq := func(x, y *Variable, r, c int) {
		xv := x.At(r, c)
		if xv >= cp.min && xv <= cp.max {
			y.Data.Set(r, c, 1)
		} else {
			y.Data.Set(r, c, 0)
		}
	}
	gy := gys[0]
	x := is[0]
	mask := &Variable{Data: nd.LikeZeros(x.Data)}
	_where(x, mask, eq)
	gx := Mul(gy, mask)
	return []*Variable{gx}
}

func Max(x *Variable, axis interface{}) *Variable {
	f := NewFunction(&maxFunc{axis: axis})
	y := f.Run(x)
	return y
}

type maxFunc struct {
	Function
	axis interface{}
}

func (s *maxFunc) forward(is []*Variable) []*Variable {
	x := is[0]
	y := _max(x, s.axis, true)
	return []*Variable{y}
}
func (s *maxFunc) backward(is, os, gys []*Variable) []*Variable {
	eq := func(x, y *Variable, r, c int) {
		if x.At(r, c) == y.At(r, c) {
			y.Data.Set(r, c, 1)
		} else {
			y.Data.Set(r, c, 0)
		}
	}
	gy := gys[0]
	x := is[0]
	y := os[0]
	shape := _maxBackwardShape(x, s.axis)
	sp := nd.NewShape(shape[0], shape[1])
	gy = Reshape(gy, sp)
	y = Reshape(y, sp)
	y = _broadcastTo(y, x.Shape())
	mask := _where(x, y, eq)
	//gy=_broadcastTo(gy,mask.Shape())
	rst := Mul(gy, mask)
	return []*Variable{rst}
}

func Min(x *Variable, axis interface{}) *Variable {
	f := NewFunction(&minFunc{axis: axis})
	y := f.Run(x)
	return y
}

type minFunc struct {
	Function
	axis interface{}
}

func (s *minFunc) forward(is []*Variable) []*Variable {
	x := is[0]
	y := _min(x, s.axis, true)
	return []*Variable{y}
}
func (s *minFunc) backward(is, os, gys []*Variable) []*Variable {
	eq := func(x, y *Variable, r, c int) {
		if x.At(r, c) == y.At(r, c) {
			y.Data.Set(r, c, 1)
		} else {
			y.Data.Set(r, c, 0)
		}
	}
	gy := gys[0]
	x := is[0]
	y := os[0]
	shape := _maxBackwardShape(x, s.axis)
	sp := nd.NewShape(shape[0], shape[1])
	gy = Reshape(gy, sp)
	y = Reshape(y, sp)
	y = _broadcastTo(y, x.Shape())
	mask := _where(x, y, eq)
	//gy=_broadcastTo(gy,mask.Shape())
	rst := Mul(gy, mask)
	return []*Variable{rst}
}
