package numgo

import (
	"sort"

	nd "test_ai/numed"
	nt "test_ai/tensor"
)

var Backprop = true

//func NewArange(start, stop, step float64) *Variable {
//	s := ut.Arange(start, stop, step)
//	return NewVec(s...)
//}
//
//func NewZeros(r, c int) *Variable {
//	s := make([]float64, r*c)
//	return NewMat(r, c, s...)
//}

//func NewOnes(r,c int) *Variable {
//	d := make([]float64, r*c)
//	for di:= range d{
//		d[di]=1
//	}
//	return NewMat(r,c,d...)
//}

//func NewRand(r, c int) *Variable {
//	s := ut.Rand(r, c)
//	return NewMat(r, c, s...)
//}
//func NewRandN(r, c int) *Variable {
//	s := ut.RandN(r, c)
//	return NewMat(r, c, s...)
//}
//func NewRandE(r, c int) *Variable {
//	s := ut.RandE(r, c)
//	return NewMat(r, c, s...)
//}
func NewVar(d float64) *Variable {
	dv := nt.NewVar(d)
	return &Variable{Data: dv}
}
func NewVec(d ...float64) *Variable {
	dv := nt.NewVec(d...)
	return &Variable{Data: dv}
}

//func NewVecInt(d ...int) *Variable {
//	dv := nd.NewVecInt(d...)
//	return &Variable{Data: dv}
//}
func NewMat(d []float64, pos ...int) *Variable {
	dv := nt.NewData(d, pos...)
	return &Variable{Data: dv}
}

//func CopyData(v *Variable) *Variable {
//	return &Variable{Data: v.Data}
//}

func NewVariable(data *nt.Tensor, name ...string) *Variable {
	v := &Variable{}
	if len(name) > 0 {
		v = &Variable{Data: data, Name: name[0]}
	} else {
		v = &Variable{Data: data}
	}
	return v
}

type Variable struct {
	Name    string
	Data    *nt.Tensor
	Grad    *Variable
	Creator *Function
	Level   int
}

func (v *Variable) Plot(verbose bool, file string) {
	PlotDotGraph(v, verbose, file)
}
func (v *Variable) ReShape(r, c int) *Variable {
	return Reshape(v, r, c)
}
func (v *Variable) Transpose() *Variable {
	return Transpose(v)
}
func (v *Variable) ClearGrade() {
	v.Grad = nil
}
func (v *Variable) SetCreator(f *Function) {
	v.Creator = f
	v.Level = f.Level + 1
}
func (v *Variable) Unchain() {
	v.Creator = nil
}
func (v *Variable) UnchainBackward() {
	var stack []*Function
	if v.Creator != nil {
		stack = append(stack, v.Creator)
	}
	for len(stack) > 0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1] //pop
		for _, x := range f.Inputs {
			if x.Creator != nil {
				stack = append(stack, x.Creator)
				x.Unchain()
			}
		}
	}
}
func (v *Variable) Backward(retainGrad bool) {
	if v.Grad == nil {
		v.Grad = NewVariable(nt.LikeOnes(v.Data))
	}
	seen := make(map[*Function]bool)
	var stack []*Function
	if v.Creator != nil {
		stack = append(stack, v.Creator)
	}
	for len(stack) > 0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1] //pop
		var gys []*Variable
		for _, output := range f.Outputs {
			gys = append(gys, output.Grad)
		}
		gxs := f.Back(gys...)
		//这里如果有两个输入，则两个输入都会加入func列表
		for xi, x := range f.Inputs {
			if x.Grad == nil {
				x.Grad = gxs[xi]
			} else {
				x.Grad = Add(x.Grad, gxs[xi])
			}
			if x.Creator != nil && !seen[x.Creator] {
				seen[x.Creator] = true
				stack = append(stack, x.Creator)
				sort.Slice(stack, func(i, j int) bool {
					return stack[i].Level < stack[j].Level
				})
			}
		}
		if !retainGrad {
			for _, o := range f.Outputs {
				o.Grad = nil
			}
		}
	}
}
func (v *Variable) Print(name string) {
	if name != "" {
		v.Data.Print(name)
	} else {
		v.Data.Print(v.Name)
	}
}
func (v *Variable) Sprint(name string) string {
	rest := ""
	if name != "" {
		rest = v.Data.Sprint(name)
	} else {
		rest = v.Data.Sprint(v.Name)
	}
	return rest
}

//func (v *Variable) At(r, c int) float64 {
//	return v.Data.Get(r, c)
//}

//func (v *Variable) DataType() string {
//	return fmt.Sprintf("%T", v.Data.At(0, 0))
//}
//func (v *Variable) Shape() *nd.Shape {
//	return nd.NewShape(v.Data.Dims())
//}
//func (v *Variable) T() *Variable {
//	return Transpose(v)
//}
//func (v *Variable) Rows(rs ...int) *Variable {
//	return &Variable{Data: v.Data.Rows(rs...)}
//}
//func (v *Variable) Var() float64 {
//	return v.Data.Var()
//}
//func (v *Variable) Mean() float64 {
//	return v.Data.Mean().Var()
//}

// NewFunction Function
func NewFunction(f IFunc) Function {
	return Function{Func: f}
}

type IFunc interface {
	forward(i []*Variable) []*Variable
	backward(i, o, dy []*Variable) []*Variable
}
type Function struct {
	Func            IFunc
	Inputs, Outputs []*Variable
	Level           int
}

func (f *Function) Run(ix ...*Variable) *Variable {
	outputs := f.Func.forward(ix)
	if Backprop {
		max := MaxLevel(ix)
		f.Level = max
		f.Inputs = ix
		for _, o := range outputs {
			o.SetCreator(f)
		}
		f.Outputs = outputs
	}
	return outputs[0]
}
func (f *Function) Back(ig ...*Variable) []*Variable {
	//todo 这里和书中不同，特殊考虑一下，要研究对错，加法在反向传播时如果没有广播，则原封不的传递上游的Grade 而一开始Grade没有Creator，会造成传递的远端的计算图没有creator，plot后只有一个变量
	//for _,g :=range ig{
	//	if g.Creator==nil{
	//		g.SetCreator(f)
	//	}
	//}
	b := f.Func.backward(f.Inputs, f.Outputs, ig)
	return b
}

//Pow Variable
func Pow(x *Variable, c int) *Variable {
	f := NewFunction(&powFunc{C: c})
	return f.Run(x)
}

type powFunc struct {
	Function
	C int
}

func (s *powFunc) forward(i []*Variable) []*Variable {
	x := i[0]
	//x.Data.Pow(s.C)
	o := nt.Pow(x.Data, s.C)
	return []*Variable{{Data: o}}
}
func (s *powFunc) backward(i, o, gy []*Variable) []*Variable {
	x := i[0]
	c := NewVariable(nt.NewVar(float64(s.C)))
	gx := Mul(Mul(Pow(x, s.C-1), gy[0]), c)
	return []*Variable{gx}
}

//Neg Variable
func Neg(x interface{}) *Variable {
	f := NewFunction(&negFunc{})
	return f.Run(AsVar(x))
}

type negFunc struct {
	Function
}

func (e *negFunc) forward(i []*Variable) []*Variable {
	x := i[0]
	//x.Data.Neg()
	o := nt.Neg(x.Data)
	return []*Variable{{Data: o}}
}
func (e *negFunc) backward(i, o, gy []*Variable) []*Variable {
	ngy := Neg(gy[0])
	return []*Variable{ngy}
}

//Add Variable
func Add(x0, x1 interface{}) *Variable {
	f := NewFunction(&addFunc{})
	y := f.Run(AsVar(x0), AsVar(x1))
	return y
}

type addFunc struct {
	Function
}

func (a *addFunc) forward(ix []*Variable) []*Variable {
	x0, x1 := ix[0], ix[1]
	o := nt.Add(x0.Data, x1.Data)
	return []*Variable{{Data: o}}
}
func (a *addFunc) backward(i, o, gy []*Variable) []*Variable {
	gx0, gx1 := gy[0], gy[0]
	x0, x1 := i[0], i[1]
	gx0, gx1 = _checkSumTo(x0.Data.Shape(), x1.Data.Shape(), gx0, gx1)
	return []*Variable{gx0, gx1}
}

//Sub Variable
func Sub(x0, x1 interface{}) *Variable {
	f := NewFunction(&subFunc{})
	y := f.Run(AsVar(x0), AsVar(x1))
	return y
}

type subFunc struct {
	Function
}

func (a *subFunc) forward(ix []*Variable) []*Variable {
	x0, x1 := ix[0], ix[1]
	o := nt.Sub(x0.Data, x1.Data)
	return []*Variable{{Data: o}}
}
func (a *subFunc) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	x0, x1 := i[0], i[1]
	gx0, gx1 := g, Neg(g)
	gx0, gx1 = _checkSumTo(x0.Data.Shape(), x1.Data.Shape(), gx0, gx1)
	return []*Variable{gx0, gx1}
}

//Mul Variable
func Mul(x0, x1 interface{}) *Variable {
	f := NewFunction(&mulFunc{})
	y := f.Run(AsVar(x0), AsVar(x1))
	return y
}

type mulFunc struct {
	Function
	x0s *nd.Shape
	x1s *nd.Shape
}

func (m *mulFunc) forward(ix []*Variable) []*Variable {
	x0, x1 := ix[0], ix[1]
	o := nt.Mul(x0.Data, x1.Data)
	return []*Variable{{Data: o}}
}
func (m *mulFunc) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	x0, x1 := i[0], i[1]
	gx0 := Mul(x1, g)
	gx1 := Mul(x0, g)
	gx0, gx1 = _checkSumTo(x0.Data.Shape(), x1.Data.Shape(), gx0, gx1)
	return []*Variable{gx0, gx1}
}

//Div Variable
func Div(x0, x1 interface{}) *Variable {
	f := NewFunction(&divFunc{})
	y := f.Run(AsVar(x0), AsVar(x1))
	return y
}

type divFunc struct {
	Function
}

func (d *divFunc) forward(ix []*Variable) []*Variable {
	x0, x1 := ix[0], ix[1]
	o := nt.Div(x0.Data, x1.Data)
	return []*Variable{{Data: o}}
}
func (d *divFunc) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	x0, x1 := i[0], i[1]
	gx0 := Div(g, x1)
	gx1 := Mul(g, Div(Neg(x0), Mul(x1, x1)))
	gx0, gx1 = _checkSumTo(x0.Data.Shape(), x1.Data.Shape(), gx0, gx1)
	return []*Variable{gx0, gx1}
}

func MaxLevel(ivs []*Variable) int {
	max := 0
	for _, v := range ivs {
		if v != nil && v.Level > max {
			max = v.Level
		}
	}
	return max
}
