package numgo

import (
	nt "test_ai/tensor"
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
	y := nt.Sin(x.Data)
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
	y := nt.Cos(x.Data)
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
	y := nt.Tanh(x.Data)
	return []*Variable{{Data: y}}
}
func (s *tanhFunc) backward(_, o, gy []*Variable) []*Variable {
	y := o[0]
	g := gy[0]
	gx := Mul(g, Neg(Sub(nt.NewVar(1), Mul(y, y))))
	return []*Variable{gx}
}

func Reshape(x *Variable, s ...int) *Variable {
	f := NewFunction(&reshapeFunc{S: s})
	y := f.Run(x)
	return y
}

type reshapeFunc struct {
	Function
	S  []int
	Xs []int
}

func (s *reshapeFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	s.Xs = x.Data.Shape()
	return []*Variable{{Data: x.Data.Reshape(s.S...)}}
}
func (s *reshapeFunc) backward(_, _, gy []*Variable) []*Variable {
	return []*Variable{Reshape(gy[0], s.Xs...)}
}

func Transpose(x *Variable, axis []int) *Variable {
	f := NewFunction(&transposeFunc{axis: axis})
	y := f.Run(x)
	return y
}

type transposeFunc struct {
	Function
	axis []int
}

func (t *transposeFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	y := x.Data.Permute(t.axis...)
	return []*Variable{{Data: y}}
}
func (t *transposeFunc) backward(_, _, gy []*Variable) []*Variable {
	y := gy[0]
	rs := make([]int, len(t.axis))
	copy(rs, t.axis)
	ut.Reverse(rs)
	return []*Variable{Transpose(y, rs)}
}

func Exp(x ...*Variable) *Variable {
	f := NewFunction(&expFunc{})
	return f.Run(x...)
}

type expFunc struct {
	Function
}

func (e *expFunc) forward(i []*Variable) []*Variable {
	x := i[0]
	return []*Variable{{Data: nt.Exp(x.Data)}}
}
func (e *expFunc) backward(_, o, gy []*Variable) []*Variable {
	y := o[0]
	g := gy[0]
	gx := Mul(g, y)
	return []*Variable{gx}
}

func Sum(x *Variable, keepDim bool, axis ...int) *Variable {
	f := NewFunction(&sumFunc{keepDim: keepDim, axis: axis})
	y := f.Run(x)
	return y
}

type sumFunc struct {
	Function
	keepDim bool
	axis    []int
	Xs      []int
}

func (s *sumFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	s.Xs = x.Data.Shape()
	y := x.Data.Sum(s.keepDim, s.axis...)
	return []*Variable{{Data: y}}
}
func (s *sumFunc) backward(i, o, gy []*Variable) []*Variable {
	gx := BroadCastTo(gy[0], s.Xs...)
	return []*Variable{gx}
}

func SumTo(x *Variable, s ...int) *Variable {
	f := NewFunction(&sumToFunc{S: s})
	y := f.Run(x)
	return y
}

type sumToFunc struct {
	Function
	S  []int
	Xs []int
}

func (s *sumToFunc) forward(ix []*Variable) []*Variable {
	x := ix[0]
	s.Xs = x.Data.Shape()
	y := nt.SumTo(x.Data, s.S)
	return []*Variable{{Data: y}}
}
func (s *sumToFunc) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	gx := BroadCastTo(g, s.Xs...)
	return []*Variable{gx}
}

func BroadCastTo(x *Variable, s ...int) *Variable {
	f := NewFunction(&broadcastToFunc{S: s})
	y := f.Run(x)
	return y
}

type broadcastToFunc struct {
	Function
	S  []int
	Xs []int
}

func (s *broadcastToFunc) forward(ix []*Variable) []*Variable {
	s.Xs = ix[0].Data.Shape()
	y := nt.BroadcastTo(ix[0].Data, s.S...)
	return []*Variable{{Data: y}}
}
func (s *broadcastToFunc) backward(i, o, gy []*Variable) []*Variable {
	gx := gy[0]
	gx = SumTo(gy[0], s.Xs...)
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
	y := nt.Dot(x.Data, W.Data)
	return []*Variable{{Data: y}}
}
func (s *matMulFunc) backward(i, o, gy []*Variable) []*Variable {
	x, W := i[0], i[1]
	g := gy[0]
	gx := Matmul(g, AsVar(W.Data.T()))
	gW := Matmul(AsVar(x.Data.T()), g)
	return []*Variable{gx, gW}
}

type linearFunc struct {
	Function
}

func (l *linearFunc) forward(ix []*Variable) []*Variable {
	x, W := ix[0], ix[1]
	y := nt.Dot(x.Data, W.Data)

	var b *Variable
	if len(ix) > 2 {
		b = ix[2]
	}
	if b != nil {
		y = nt.Add(y, b.Data)
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
		gb = SumTo(gy, b.Data.Shape()...)
	}
	gx := Matmul(gy, AsVar(W.Data.T()))
	gW := Matmul(AsVar(x.Data.T()), gy)
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
	y := nt.Add(nt.Mul(nt.Tanh(nt.Mul(x.Data, nt.NewVar(0.5))), nt.NewVar(0.5)), nt.NewVar(0.5))
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
	diff := nt.Sub(x0.Data, x1.Data)
	diffLen := diff.Shape()[0]
	//不保持形状，计算所有合计数值
	y := nt.Pow(diff, 2).Sum(false).Var() / float64(diffLen)
	v := &Variable{Data: nt.NewVar(y)}
	return []*Variable{v}
}
func (m *meanSquaredError) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x0, x1 := is[0], is[1]
	diff := Sub(x0, x1)
	gx0 := Mul(Mul(gy, diff), Div(2, diff.Data.Shape()[0]))
	gx1 := Neg(gx0)
	return []*Variable{gx0, gx1}
}

func MeanSquaredError(x0, x1 *Variable) *Variable {
	f := NewFunction(&meanSquaredError{})
	return f.Run(x0, x1)
}
func Softmax1d(x *Variable) *Variable {
	y := nt.Exp(x.Data)
	//todo 这里改成tersor后要核查一下是否是keepdim=true
	sumY := y.Sum(true, 1)
	return Div(y, sumY)
}

func Softmax_cross_entropy_simple(x *Variable, t []int) float64 {
	N := x.Data.Shape()[0]
	p := Softmax1d(x)
	p = Clip(p, 1e-15, 1.0)
	p = Log(p)
	p = Getitem(p, ut.ArangeInt(0, N, 1), t)
	sum := Sum(p, false)
	y := -1 * sum.Data.Get(0) / float64(N)
	return y
}

func Accuracy(y interface{}, t []float64) *Variable {
	yv := AsVar(y)
	tv := NewVariable(nt.NewVec(t...))
	pred := yv.Data.ArgMax(1, true).Reshape(tv.Data.Dims())
	mask := nt.ApplyPos(pred, func(pos []int, v float64) float64 {
		if v == tv.Data.Get(pos...) {
			return 1.0
		} else {
			return 0.0
		}
	})
	return NewVariable(mask.Mean(false))
}

func Dropout(xi interface{}, dropRatio float64) *Variable {
	x := AsVar(xi)
	if Backprop { //for train
		randD := nt.NewRand(x.Data.Shape()...)
		mask := nt.ApplyPos(randD, func(pos []int, v float64) float64 {
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

func SoftmaxCrossEntroy(x interface{}, t []float64) *Variable {
	xv := AsVar(x)
	//todo 先在这里转换城整数，如果需要就从根本上把dataload返回的标签改成整数
	f := NewFunction(&softmaxCrossEntropy{t: ut.ToInt(t)})
	y := f.Run(xv)
	return y
}

type softmaxCrossEntropy struct {
	Function
	t []int
}

func (s *softmaxCrossEntropy) forward(is []*Variable) []*Variable {
	x := is[0]
	N := x.Data.Shape()[0]
	logZ := _logsumexp(x, 1)
	logP := nt.Sub(x.Data, logZ.Data)                          //Sub(x,logZ)
	logP = logP.SliceSel(0, false, ut.ArangeInt(0, N, 1), s.t) // SelRowCol(logP.Data,s.t...) //Getitem(logP,ut.ArangeInt(0,N,1),s.t)
	sum := -1 * logP.Sum(false).Get(0)
	y := sum / float64(N)
	return []*Variable{AsVar(y)}
}
func (s *softmaxCrossEntropy) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x := is[0]
	xs := x.Data.Shape()
	N, ClsNum := xs[0], xs[1]
	gy = Mul(gy, 1.0/float64(N))
	y := Softmax(x, 1)
	//to one hot
	eyes := _eyes(ClsNum)
	tOneHot := OneHot(eyes, s.t)
	y = Mul(Sub(y, tOneHot), gy)
	return []*Variable{y}
}

func Softmax(x *Variable, axis int) *Variable {
	f := NewFunction(&softmax{axis: axis})
	y := f.Run(x)
	return y
}

type softmax struct {
	Function
	axis int
}

func (s *softmax) forward(is []*Variable) []*Variable {
	x := is[0]
	y := nt.Sub(x.Data, nt.Max(x.Data, true, s.axis))
	y = nt.Exp(y)
	ym := nt.Sum(y, true, s.axis)
	y = nt.Div(y, ym)
	return []*Variable{{Data: y}}
}
func (s *softmax) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	y := os[0]
	gx := Mul(y, gy)
	sumdx := Sum(gx, true, s.axis)
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
	//y := nt.Maximum(x.Data, 0.0)
	y := nt.Maximum(x.Data, nt.NewVar(0.0))
	return []*Variable{{Data: y}}
}
func (s *reLU) backward(is, os, gys []*Variable) []*Variable {
	x := is[0]
	mask := nt.ApplyPos(x.Data, func(pos []int, v float64) float64 {
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
		y = append(y, x.Data.Get(r, c))
	}
	return []*Variable{{Data: nt.NewVec(y...)}}
}

func (s *getItem) backward(is, os, gys []*Variable) []*Variable {
	//x:=is[0]

	return []*Variable{}
}

type GetItemGrad struct {
	Function
}

func Log(x *Variable) *Variable {
	x.Data.Log()
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
	y := nt.Clip(x.Data, cp.min, cp.max)
	return []*Variable{{Data: y}}
}
func (cp *clipFunc) backward(is, os, gys []*Variable) []*Variable {
	eq := func(x, y *Variable, r, c int) {
		xv := x.Data.Get(r, c)
		if xv >= cp.min && xv <= cp.max {
			y.Data.Set(1, r, c)
		} else {
			y.Data.Set(0, r, c)
		}
	}
	gy := gys[0]
	x := is[0]
	mask := &Variable{Data: nt.LikeZeros(x.Data)}
	_where(x, mask, eq)
	gx := Mul(gy, mask)
	return []*Variable{gx}
}

func Max(x *Variable, keepDims bool, axis ...int) *Variable {
	f := NewFunction(&maxFunc{axis: axis, keepDims: keepDims})
	y := f.Run(x)
	return y
}

type maxFunc struct {
	Function
	axis     []int
	keepDims bool
}

func (s *maxFunc) forward(is []*Variable) []*Variable {
	x := is[0]
	y := nt.Max(x.Data, s.keepDims, s.axis...)
	return []*Variable{{Data: y}}
}
func (s *maxFunc) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x := is[0]
	y := os[0]
	shape := _maxBackwardShape(x, s.axis)
	gy = Reshape(gy, shape...)
	y = Reshape(y, shape...)
	//y = _broadcastTo(y, x.Data.Shape())
	//mask := _where(x, y, eq)
	////gy=_broadcastTo(gy,mask.Shape())
	mask := nt.ApplyBroadcast(x.Data, y.Data, func(idx int, v, ov float64) float64 {
		if v == ov {
			return 1.0
		} else {
			return 0.0
		}
	})
	rst := Mul(gy, mask)
	return []*Variable{rst}
}

func Min(x *Variable, keepDims bool, axis ...int) *Variable {
	f := NewFunction(&minFunc{axis: axis, keepDims: keepDims})
	y := f.Run(x)
	return y
}

type minFunc struct {
	Function
	axis     []int
	keepDims bool
}

func (s *minFunc) forward(is []*Variable) []*Variable {
	x := is[0]
	y := nt.Min(x.Data, s.keepDims, s.axis...)
	return []*Variable{{Data: y}}
}
func (s *minFunc) backward(is, os, gys []*Variable) []*Variable {
	gy := gys[0]
	x := is[0]
	y := os[0]
	shape := _maxBackwardShape(x, s.axis)
	gy = Reshape(gy, shape...)
	y = Reshape(y, shape...)
	//y = _broadcastTo(y, x.Data.Shape())
	//mask := _where(x, y, eq)
	//gy=_broadcastTo(gy,mask.Shape())
	mask := nt.ApplyBroadcast(x.Data, y.Data, func(idx int, v, ov float64) float64 {
		if v == ov {
			return 1.0
		} else {
			return 0.0
		}
	})
	rst := Mul(gy, mask)
	return []*Variable{rst}
}
